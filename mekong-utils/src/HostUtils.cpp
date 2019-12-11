#include "Mekong-Utils.h"

#include <llvm/Support/Casting.h>
// instead of:
//#include <llvm/IR/TypeBuilder.h>
// adapt with llvm version?
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include "llvm/Support/VersionTuple.h"

using namespace llvm;
using namespace std;

bool launchFinder(BasicBlock *block, void *launchCallPtr) {
  CallBase **ptr = (CallBase **)launchCallPtr;

  for (Instruction &inst : *block) {

    // If inst is a call inst look for cuda funktions
    if (CallBase *ci = dyn_cast_or_null<CallBase>(&inst)) {
      string name = ci->getCalledFunction()->getName();
      if (name == "cudaSetupArgument")
        continue;
      if (name == "cudaLaunch") {
        *ptr = ci;
        return true;
        // Any other call could be the kernel wrapper
        // Check if cudaLaunch is called from the possible kernel wrapper
        // function
      } else {

        Function *cudaLaunch =
            block->getParent()->getParent()->getFunction("cudaLaunch");
        for (Value *launchCallVal : cudaLaunch->users()) {
          if (CallBase *launchCallBase =
                  dyn_cast_or_null<CallBase>(launchCallVal)) {
            if (launchCallBase->getFunction() == ci->getCalledFunction()) {
              *ptr = launchCallBase;
              return true;
            }
          }
        } // end for (Value* launchCallVal : cudaLaunch->users())

      } // end else
    }   // end if (CallBase *ci = dync_cast_or_null<CallBase>(&inst))
    if (InvokeInst *invi = dyn_cast_or_null<InvokeInst>(&inst)) {
      // Check if cudaLaunch is called from the possible kernel wrapper function
      Function *cudaLaunch =
          block->getParent()->getParent()->getFunction("cudaLaunch");
      for (Value *launchCallVal : cudaLaunch->users()) {
        if (CallBase *launchCallBase =
                dyn_cast_or_null<CallBase>(launchCallVal)) {
          if (launchCallBase->getFunction() == invi->getCalledFunction()) {
            *ptr = launchCallBase;
            return true;
          }
        }
      } // end for (Value* launchCallVal : cudaLaunch->users())

    } // end if (CallBase *ci = dync_cast_or_null<CallBase>(&inst))

  } // end for (Insturction &inst : *block)

  return false;
}

namespace mekong {

///===------------------------------------------------------------------===//
///                           Analysis Functions
///===------------------------------------------------------------------===//

bool usesNewKernelLaunch(llvm::Module &m) {
  return m.getSDKVersion() >= VersionTuple(9,2);
}

// Finds all functions which are being used to register a cuda kernel with the cuda driver
void getKernelHandles(llvm::Module &m, std::vector<llvm::Function*> &handles) {
  Function *registerFunction = m.getFunction("__cudaRegisterFunction");
  for (auto *user : registerFunction->users()) {
    CallBase *callBase = dyn_cast_or_null<CallBase>(user);
    if (callBase != nullptr) {
      Value *val = (callBase->getArgOperand(1)->stripPointerCastsAndAliases());
      // 2nd argument is the function pointer which acts as handle
      // strip all casts etc to get the function itself and not an intermediate value
      Function* handle = dyn_cast_or_null<Function>(val);
      if (handle != nullptr) {
        handles.push_back(handle);
      }
    }
  }
  return;
}

// Finds the instructions that call the kernel launch wrapper function (klFun)
// in this function cudaLaunch or cudaLaunchKernel will be called
// this function will not return the call sites if the wrapper function is already inlined
void getKernelLaunchSites(llvm::Function *klFun, std::vector<llvm::CallBase*> &callSites) {
  for (auto *user : klFun->users()) {
    CallBase *callBase = dyn_cast_or_null<CallBase>(user);
    // user must be callbase and call the kernel launch function
    if (callBase != nullptr && (callBase->getCalledFunction() == klFun)) {
      callSites.push_back(callBase);
    }
  }
  return;
}


void getKernelArguments(llvm::CallBase *kernelLaunchSite, std::vector<llvm::Value*> &args) {
  for (auto &op : kernelLaunchSite->operands()) {
    // Make sure not to inclide the kernel launch wrapper
    Function *fun = dyn_cast_or_null<Function>(&op);
    if ( fun != nullptr and fun == kernelLaunchSite->getCalledFunction() )
      continue;
    // also skip invoke operands which are basicblocks
    if (dyn_cast_or_null<BasicBlock>(&op) != nullptr)
      continue;

    args.push_back(op);
  }
  // get arguments regarding launch configuration
  return;  
}

CallBase* getKernelConfigCall(llvm::Module &m, llvm::CallBase *kernelLaunchSite) {
  Function *confFunc = nullptr;
  
  if (usesNewKernelLaunch(m)) {
    confFunc = m.getFunction("__cudaPushCallConfiguration");
  } else {
    confFunc = m.getFunction("cudaConfigureCall");
  }

  assert(confFunc != nullptr);

  CallBase *confCall = nullptr;
  int i = 3;
  BasicBlock *currentBlock = kernelLaunchSite->getParent()->getSinglePredecessor();
  while (confCall == nullptr) {
    assert(currentBlock != nullptr);
    for (Instruction &inst : *currentBlock) {
      CallBase *ci = dyn_cast_or_null<CallBase>(&inst);
      if (ci != nullptr && ci->getCalledFunction() == confFunc) {
        confCall = ci;
      }
    }
    --i;
    assert(i>0 && "backtrack exceeded max number of basic blocks to rewind. Configuration call not found!");
    currentBlock = currentBlock->getSinglePredecessor();
  }

  assert(confCall != nullptr);

  return confCall;
}

void getKernelLaunchConfig(llvm::Module &m, llvm::CallBase *kernelLaunchSite, std::vector<llvm::Value*> &config) {
  CallBase *confCall = getKernelConfigCall(m, kernelLaunchSite);
  
  for (auto &val : confCall->operands()) {
    // only the arguments are wanted not the function itself
    if (dyn_cast_or_null<Function>(&val) != nullptr)
      continue;
    // discard basic blocks in case the callbase is an invoke instruction
    if (dyn_cast_or_null<BasicBlock>(&val) != nullptr)
      continue;
    config.push_back(val);
  }
  return;
}

void getKernelLaunches(
    llvm::Module &m,
    std::vector<std::pair<llvm::CallBase *, llvm::CallBase *>> &kernelLaunch) {
  Function *cudaConfigureCall = m.getFunction("cudaConfigureCall");
  Function *cudaLaunch = m.getFunction("cudaLaunch");

  // If both function are not found assume there are no kernel launches
  if (cudaConfigureCall == nullptr and cudaLaunch == nullptr) {
    errs() << "No cudaConfigure and no cudaLaunch found. Assume there are no "
              "kernel launches\n";
    return;
  }

  assert(cudaConfigureCall != nullptr &&
         "function could not be found in module!");
  assert(cudaLaunch != nullptr && "function could not be found in module!");

  for (Value *confCallVal : cudaConfigureCall->users()) {
    // CallBase *confCallBase = dyn_cast_or_null<CallBase>(confCallVal);
    CallBase *confCallBase = dyn_cast_or_null<CallBase>(confCallVal);
    InvokeInst *confInvokeInst = dyn_cast_or_null<InvokeInst>(confCallVal);
    assert((confCallBase != nullptr or confInvokeInst != nullptr) &&
           "Could not determine inst. type of cudaConfigure call");
    BasicBlock *parent = nullptr;
    if (confInvokeInst != nullptr) {
      parent = confInvokeInst->getNormalDest();
    } else {
      parent = confCallBase->getParent();
    }

    // Assumption: There are exactly two successors and one is the one if
    // cudaConfigureCall succeeds
    BasicBlock *next = parent->getTerminator()->getSuccessor(0);
    BasicBlock *failBlock = parent->getTerminator()->getSuccessor(1);

    if (next->getName().find("kcall.configok") == StringRef::npos) {
      assert(failBlock->getName().find("kcall.configok") != StringRef::npos &&
             "Could not select next block!");
      swap(next, failBlock);
    }

    CallBase *launchCall = nullptr;
    visitNodes(next, failBlock, &launchCall, launchFinder);
    assert(launchCall != nullptr && "Could not find cudaLaunch");
    assert(confCallBase != nullptr && "Fuck");
    kernelLaunch.push_back({confCallBase, launchCall});
  }
}

/// Returns arguments for pre CUDA 7.0 kernel launches
/// Assumes that the kernel-wrapper was not inlined!
void getLaunchArguments(llvm::Module &m, llvm::CallBase *configureCall,
                        llvm::CallBase *launchCall,
                        std::vector<llvm::Value *> &args) {
  // Check KernelWrapper Function was inlined
  assert(configureCall->getFunction() != launchCall->getFunction() &&
         "Kernel wrapper is inlined. Cannot find kernel arguments!");

  BasicBlock *launchBlock = nullptr;
  BranchInst *bi =
      dyn_cast_or_null<BranchInst>(configureCall->getParent()->getTerminator());
  if (bi != nullptr) {
    assert(bi != nullptr && bi->getNumSuccessors() == 2);
    launchBlock = bi->getSuccessor(0);
    if (launchBlock->getName().find("kcall.configok") == StringRef::npos)
      launchBlock = bi->getSuccessor(1);
  } else {
    InvokeInst *inv = dyn_cast_or_null<InvokeInst>(
        configureCall->getParent()->getTerminator());
    assert(inv != nullptr);
    BasicBlock *dest = inv->getNormalDest();
    launchBlock = dest->getTerminator()->getSuccessor(0);
    if (launchBlock->getName().find("kcall.configok") == StringRef::npos)
      launchBlock = dest->getTerminator()->getSuccessor(1);
  }

  assert(launchBlock != nullptr);

  for (auto &inst : *launchBlock) {
    // Check for Instruction Calling the Kernel launch Wrapper
    CallBase *ci = dyn_cast_or_null<CallBase>(&inst);

    if (ci != nullptr &&
        (ci->getCalledFunction() == launchCall->getFunction())) {
      for (auto &val : ci->operands()) {
        // only the arguments are wanted not the function itself
        if (dyn_cast_or_null<Function>(&val) != nullptr)
          continue;
        // also skip invoke operands which are basicblocks
        if (dyn_cast_or_null<BasicBlock>(&val) != nullptr)
          continue;
        // val->dump();
        args.push_back(val);
      }
      continue;
    }
  }
}

llvm::Function *getCudaSynchronizeStream(llvm::Module &m) {
  LLVMContext &ctx = m.getContext();
  FunctionType *cudaSyncStream_ft = FunctionType::get(
      Type::getInt32Ty(ctx),
      {m.getTypeByName("struct.CUstream_st")->getPointerTo()}, false);
  return dyn_cast<Function>(
      m.getOrInsertFunction("cudaStreamSynchronize", cudaSyncStream_ft)
          .getCallee());
}

void getGridConfig(llvm::CallBase *call, llvm::Value *(&arguments)[4]) {
  assert(call->getCalledFunction()->getName() == "cudaLaunchKernel");
  arguments[0] = call->getArgOperand(1);
  arguments[1] = call->getArgOperand(2);
  arguments[2] = call->getArgOperand(3);
  arguments[3] = call->getArgOperand(3);
}

///===------------------------------------------------------------------===//
///                        Transformation Functions
///===------------------------------------------------------------------===//

void registerKernel(llvm::Module &m, const std::string name,
                    llvm::Function *kernelWrapper) {
  LLVMContext &ctx = m.getContext();

  Function *registerGlobals = m.getFunction("__cuda_register_globals");

  assert(registerGlobals != nullptr &&
         "Could not find __cuda_register_globals!");
  Function *registerFunction = m.getFunction("__cudaRegisterFunction");
  assert(registerFunction != nullptr &&
         "Could not find __cudaRegisterFunction!");

  // find cuda binary handle
  Value *bin_handle = nullptr;

  if (registerGlobals != nullptr) {
    bin_handle = &*registerGlobals->arg_begin();

  } else { // due to the asser this should never be reached
    registerGlobals = m.getFunction("__cuda_module_ctor");

    for (auto &bb : *registerGlobals) {
      for (auto &inst : bb) {
        if (CallBase *ci = dyn_cast_or_null<CallBase>(&inst)) {
          if (ci->getCalledFunction()->getName() == "__cudaRegisterFatBinary") {
            //                                       __cudaRegisterFatBinary
            bin_handle = ci;
          }
        }
      }
    }
  }

  assert(bin_handle != nullptr && "No cuda binary handle found!");

  IRBuilder<> builder(ctx);
  builder.SetInsertPoint(&registerGlobals->back().back());

  Value *wrapper_casted =
      builder.CreateBitCast(kernelWrapper, builder.getInt8PtrTy());
  Value *globalKernelNameString = builder.CreateGlobalStringPtr(name);
  Value *null = ConstantPointerNull::get(builder.getInt8PtrTy());
  Value *int32null =
      ConstantPointerNull::get(builder.getInt32Ty()->getPointerTo());
  vector<Value *> registerFunctionsArgs = {bin_handle,
                                           wrapper_casted,
                                           globalKernelNameString,
                                           globalKernelNameString,
                                           builder.getInt32(-1),
                                           null,
                                           null,
                                           null,
                                           null,
                                           int32null};

  CallBase *errorCode =
      builder.CreateCall(registerFunction, registerFunctionsArgs);
  // mekong::callPrintf(m, "RegisterFunction: %d\n",
  // errorCode)->insertAfter(errorCode);
}

/// Creates a wrapper for a cuda kernel in module m with the given name
/// The functiontype is the the function signature of the kernel
Function *createKernelWrapper(llvm::Module &m, const std::string name,
                              FunctionType *ft) {
  LLVMContext &ctx = m.getContext();

  IRBuilder<> builder(ctx);
  FunctionType *launchType = FunctionType::get(
      builder.getInt32Ty(),
      {builder.getInt8PtrTy(), builder.getInt64Ty(), builder.getInt32Ty(),
       builder.getInt64Ty(), builder.getInt32Ty(),
       builder.getInt8PtrTy()->getPointerTo(), builder.getInt64Ty(),
       m.getTypeByName("struct.CUstream_st")->getPointerTo()},
      false);
  // alternativ:
  // m.getFunction("cudaConfigureCall")->Type()-><LetzterTypderArgumente>

  Function *launchKernel = dyn_cast<Function>(
      m.getOrInsertFunction("cudaLaunchKernel", launchType).getCallee());

  // Same convention like the cuda api function: grid, and block sizes are i32
  // variables and the first two are merged to one i64
  vector<Type *> wrapperParams = {
      builder.getInt64Ty(),
      builder.getInt32Ty(),
      builder.getInt64Ty(),
      builder.getInt32Ty(),
      builder.getInt64Ty(),
      m.getTypeByName("struct.CUstream_st")->getPointerTo()};

  // add kernel parameters
  for (auto *param : ft->params())
    wrapperParams.push_back(param);

  FunctionType *wrapperType =
      FunctionType::get(builder.getInt32Ty(), wrapperParams, false);

  Function *wrapper =
      dyn_cast<Function>(m.getOrInsertFunction(name, wrapperType).getCallee());

  BasicBlock *entry = BasicBlock::Create(ctx, "entry", wrapper, nullptr);

  builder.SetInsertPoint(entry);

#if 0
  // Check if upper half of the GridDim which corresponds to gridY is zero and in that case increase it by one
  Value* gridY = builder.CreateAnd(builder.getInt64(0xFFFFFFFF00000000), wrapper->arg_begin()+0);

  Value* isGridYZero = builder.CreateICmpEQ(builder.getInt64(0x0000000000000000), gridY);

  BasicBlock* gridYZero = BasicBlock::Create(ctx, "gridY.Zero", wrapper, nullptr);

  BasicBlock* gridYOk = BasicBlock::Create(ctx, "gridY.ok", wrapper, nullptr);

  builder.CreateCondBr(isGridYZero, gridYZero, gridYOk);

  builder.SetInsertPoint(gridYZero);

  Value *gridXYinc = builder.CreateAnd(builder.getInt64(0x0000000100000000), wrapper->arg_begin()+0);
  builder.CreateBr(gridYOk);

  builder.SetInsertPoint(gridYOk);

  PHINode *gridXY = builder.CreatePHI(gridXYinc->getType(), 2, "");

  gridXY->addIncoming(gridXYinc, gridYZero);
  gridXY->addIncoming(wrapper->arg_begin()+0, entry);

  // Check if upper half of the BlockDim which corresponds to blockY is zero and in that case increase it by one
  Value* blockY = builder.CreateAnd(builder.getInt64(0xFFFFFFFF00000000), wrapper->arg_begin()+2);

  Value* isBlockYZero = builder.CreateICmpEQ(builder.getInt64(0x0000000000000000), blockY);

  BasicBlock* blockYZero = BasicBlock::Create(ctx, "blockY.Zero", wrapper, nullptr);

  BasicBlock* blockYOk = BasicBlock::Create(ctx, "blockY.ok", wrapper, nullptr);

  builder.CreateCondBr(isBlockYZero, blockYZero, blockYOk);

  builder.SetInsertPoint(blockYZero);

  Value *blockXYinc = builder.CreateAnd(builder.getInt64(0x0000000100000000), wrapper->arg_begin()+2);
  builder.CreateBr(blockYOk);

  builder.SetInsertPoint(blockYOk);

  PHINode *blockXY = builder.CreatePHI(blockXYinc->getType(), 2, "");

  blockXY->addIncoming(blockXYinc, blockYZero);
  blockXY->addIncoming(wrapper->arg_begin()+2, gridYOk);

  //wrapper->dump();
#endif

  // Get the 8 Arguments for cudaLaunchKernel (dim3 is translated to i64 + i32)
  // cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim,
  //                    void** args, size_t sharedMem, cudaStream_t stream )
  vector<Value *> launchKernelArgs;

  // 1. function handle
  Value *fuptr = builder.CreatePointerCast(wrapper, builder.getInt8PtrTy());
  launchKernelArgs.push_back(fuptr);

  // 2.-5. gridDim and blockDim
  auto it = wrapper->arg_begin();
  // launchKernelArgs.push_back(gridXY);  // Skip Original Grid XY  Value with
  // fixed one
  launchKernelArgs.push_back(&*it);
  ++it;
  // Grid Z
  launchKernelArgs.push_back(&*it);
  ++it;
  // launchKernelArgs.push_back(blockXY);   // Skip original block XY value with
  // fixed one
  launchKernelArgs.push_back(&*it);
  ++it;
  launchKernelArgs.push_back(&*it);
  ++it;

  Value *shm = &*(it++);    // Shared Memory
  Value *stream = &*(it++); // Stream

  // 6. pointer to argument ptr array
  Value *ptr_array = builder.CreateAlloca(
      builder.getInt8PtrTy(), builder.getInt64(ft->params().size()));

  for (int i = 0; i < ft->params().size(); ++i) {
    // Get the parameter for the kernel, which start after the grid etc
    Value *argument = wrapper->arg_begin() + 6 + i;
    Value *memptr = builder.CreateAlloca(argument->getType());
    Value *store = builder.CreateStore(argument, memptr);
    Value *strptr = builder.CreateGEP(ptr_array, builder.getInt64(i));
    strptr = builder.CreatePointerCast(
        strptr, argument->getType()->getPointerTo()->getPointerTo());
    builder.CreateStore(memptr, strptr);
  }

  launchKernelArgs.push_back(ptr_array);

  // 7. + 8.
  launchKernelArgs.push_back(shm);
  launchKernelArgs.push_back(stream);

  Value *ret = builder.CreateCall(launchKernel, launchKernelArgs);

  builder.CreateRet(ret);

  return wrapper;
}

Function *createDummyKernelWrapper(llvm::Module &m, const std::string name) {
  LLVMContext &ctx = m.getContext();

  IRBuilder<> builder(ctx);
  FunctionType *fty = FunctionType::get(builder.getVoidTy(), {}, false);

  Function *dummy =
      dyn_cast<Function>(m.getOrInsertFunction(name, fty).getCallee());

  BasicBlock *entry = BasicBlock::Create(ctx, "entry", dummy, nullptr);

  builder.SetInsertPoint(entry);
  builder.CreateRet(builder.getInt32(0));

  return dummy;
}

llvm::CallBase *replaceKernelLaunch(llvm::Module &m,
			 llvm::CallBase *kernelLaunchSite,
			 llvm::Function *replacementWrapper,
			 std::vector<llvm::Value*> &additionalArguments) {
  // gather launch config and kernel arguments
  std::vector<Value*> config;
  getKernelLaunchConfig(m, kernelLaunchSite, config);
  std::vector<Value*> kargs;
  getKernelArguments(kernelLaunchSite, kargs);
  CallBase *confCall = getKernelConfigCall(m, kernelLaunchSite);
  BasicBlock *launchBlock = kernelLaunchSite->getParent();

  LLVMContext &ctx = m.getContext();
  IRBuilder<> builder(ctx);

  // check if kernelLaunch is invoke inst and insert branch if so
  InvokeInst *kernelInv = dyn_cast_or_null<InvokeInst>(kernelLaunchSite);
  if (kernelInv != nullptr) {
    BasicBlock *branchTarget = kernelInv->getNormalDest();
    builder.SetInsertPoint(kernelLaunchSite);
    builder.CreateBr(branchTarget);
  }
    
  // remove old kernel launch
  kernelLaunchSite->eraseFromParent();

  // remove intermediate blocks between launch and config call
  BasicBlock *currentBlock = launchBlock->getSinglePredecessor();
  assert(currentBlock != nullptr);
  if (confCall->getParent() != currentBlock) {
    BasicBlock *prev = currentBlock->getSinglePredecessor();
    Instruction *termInst = prev->getTerminator();
    InvokeInst *invInst = dyn_cast_or_null<InvokeInst>(termInst);
    assert( invInst != nullptr);
    invInst->setNormalDest(launchBlock);
    currentBlock->eraseFromParent();
  }

  // Branch to Launch Block Before Configure Call
  // New code will be inserted before cudaConfigureCall an a new basic block
  BasicBlock *insertPoint = confCall->getParent();
  // New Block begins with configure call
  BasicBlock *newBlock = confCall->getParent()->splitBasicBlock(confCall);
  // Insert code before in block before configure call
  Instruction *terminator = insertPoint->getTerminator();
  builder.SetInsertPoint(terminator);
  builder.CreateBr(launchBlock);
  terminator->eraseFromParent();

  // remove configure call
  for(auto *user : confCall->users()) {
    user->dropAllReferences();
  }
  confCall->eraseFromParent();
  newBlock->eraseFromParent();
  
  // Prepare Launch Call Args
  std::vector<Value *> args;
  // append kernel launch config
  for (auto &val : config) {
    args.push_back(val);
  }
  // append kernel arguments
  for (auto &val : kargs) {
    args.push_back(val);
  }
  // append additional arguments
  for (auto &val : additionalArguments) {
    args.push_back(val);
  }



  // cast pointer to stream if types do not match
  builder.SetInsertPoint(launchBlock->getTerminator());
  FunctionType *ft = replacementWrapper->getFunctionType();
  auto params = ft->params();
  if (args[5]->getType() != params[5]) {
    args[5] = builder.CreateBitCast(args[5], params[5]);
  }
  // insert new kernel launch
  CallBase *newLaunchCall = builder.CreateCall(replacementWrapper, args);
  
  return newLaunchCall;
}

llvm::CallBase *replaceKernelCall(llvm::Module &m,
                                  llvm::CallBase *configureCall,
                                  llvm::CallBase *launchCall,
                                  llvm::Function *replacement,
                                  std::vector<Value *> &additionalArguments) {
  LLVMContext &ctx = m.getContext();
  IRBuilder<> builder(ctx);

  // New code will be inserted before cudaConfigureCall an a new basic block
  BasicBlock *insertPoint = configureCall->getParent();
  // New Block begins with configure call
  BasicBlock *newBlock =
      configureCall->getParent()->splitBasicBlock(configureCall);

  // Insert code before in block before configure call
  Instruction *terminator = insertPoint->getTerminator();
  builder.SetInsertPoint(terminator);

  std::vector<Value *> args;

  // get arguments regarding launch configuration
  for (auto &val : configureCall->operands()) {
    // only the arguments are wanted not the function itself
    if (dyn_cast_or_null<Function>(&val) != nullptr)
      continue;
    if (dyn_cast_or_null<BasicBlock>(&val) != nullptr)
      continue;
    // val->dump();
    args.push_back(val);
  }

  // Find block where kernel wrapper is called
  // TODO this should be a function as it is done in multiple places
  BasicBlock *launchBlock = nullptr;
  BranchInst *bi = dyn_cast_or_null<BranchInst>(newBlock->getTerminator());
  if (bi != nullptr) {
    assert(bi->getNumSuccessors() == 2);
    launchBlock = bi->getSuccessor(0);
    if (launchBlock->getName().find("kcall.configok") == StringRef::npos)
      launchBlock = bi->getSuccessor(1);
  } else {
    InvokeInst *inv = dyn_cast_or_null<InvokeInst>(
        configureCall->getParent()->getTerminator());
    assert(inv != nullptr);
    BasicBlock *dest = inv->getNormalDest();
    launchBlock = dest->getTerminator()->getSuccessor(0);
    if (launchBlock->getName().find("kcall.configok") == StringRef::npos)
      launchBlock = dest->getTerminator()->getSuccessor(1);
  }

  assert(launchBlock != nullptr);

  // Save the Predecessor of launchBlock for later use
  BasicBlock *preLaunchBlock = launchBlock->getSinglePredecessor();
  assert(preLaunchBlock != nullptr);

  // Get the Kernel Arguments
  getLaunchArguments(m, configureCall, launchCall, args);

  for (auto *val : additionalArguments)
    args.push_back(val);

  // Save instruction that have to be moved later
  std::vector<Instruction *> move;

  CallBase *kernelWrapperCall = nullptr;

  for (auto &inst : *launchBlock) {
    CallBase *ci = dyn_cast_or_null<CallBase>(&inst);
    // Check if the functioncall is the call to the old wrapper function
    // and don't add it to the instructions to move in that case
    if (ci != nullptr and
        ci->getCalledFunction() == launchCall->getFunction()) {
      kernelWrapperCall = ci;
      continue;
    }
    BranchInst *ti = dyn_cast_or_null<BranchInst>(&inst);
    if (ti == nullptr) {
      move.push_back(&inst);
    }
  }

  CallBase *replacementLaunchCall = builder.CreateCall(replacement, args);
  InvokeInst *inv = dyn_cast_or_null<InvokeInst>(kernelWrapperCall);

  if (inv != nullptr) {
    builder.CreateBr(inv->getNormalDest());
  } else {
    builder.CreateBr(launchBlock->getSingleSuccessor());
  }

  // Cleanup
  for (auto *inst : move) {
    inst->moveBefore(replacementLaunchCall);
  }
  terminator->eraseFromParent();
  // in case of calling the kernel wrapper with invoke, there is an additional
  // block
  if (newBlock != preLaunchBlock) {
    for (auto *user : preLaunchBlock->users())
      user->dropAllReferences();
    preLaunchBlock->eraseFromParent();
  }
  newBlock->eraseFromParent();
  launchBlock->eraseFromParent();

  return replacementLaunchCall;
}

llvm::Value *createCudaGlobalVar(llvm::Module &m, const std::string name,
                                 llvm::Type *varType) {
  LLVMContext &ctx = m.getContext();

  IRBuilder<> builder(ctx);

  Value *handle = new GlobalVariable(
      m, varType, false, GlobalValue::InternalLinkage, nullptr, name);

  Function *registerGlobals = m.getFunction("__cuda_register_globals");

  assert(registerGlobals != nullptr &&
         "No entrypoint to register cuda global variables");

  vector<Type *> cuRegVarArgs = {builder.getInt8PtrTy()->getPointerTo(),
                                 builder.getInt8PtrTy(),
                                 builder.getInt8PtrTy(),
                                 builder.getInt8PtrTy(),
                                 builder.getInt32Ty(),
                                 builder.getInt32Ty(),
                                 builder.getInt32Ty(),
                                 builder.getInt32Ty()};

  FunctionType *cuRegVarTy =
      FunctionType::get(builder.getInt32Ty(), cuRegVarArgs, false);

  Function *cuRegVar = dyn_cast_or_null<Function>(
      m.getOrInsertFunction("__cudaRegisterVar", cuRegVarTy).getCallee());
  assert(cuRegVar != nullptr && "Could not get Function __cudaRegisterVar");

  builder.SetInsertPoint(&(registerGlobals->back().back()));

  Value *castHandle = builder.CreateBitCast(handle, builder.getInt8PtrTy());

  Value *strVal = builder.CreateGlobalStringPtr(name);

  builder.CreateCall(cuRegVar,
                     {&*registerGlobals->arg_begin(), castHandle, strVal,
                      strVal, builder.getInt32(0), builder.getInt32(4),
                      builder.getInt32(0), builder.getInt32(0)});

  return handle;
}

#if 0
void registerGlobalVar(Module &m, string name, Type* type, GlobalVariable *&gv) {
  LLVMContext &ctx = m.getContext();
  gv = new GlobalVariable(m, type, false, GlobalValue::ExternalLinkage,
                          nullptr, name, nullptr, GlobalVariable::NotThreadLocal,
                          1, true);
  gv->setAlignment(4);
}

void loadGlobalVar(Function *kernel, GlobalVariable *gv, Value *&val) {
  Instruction *insertPoint = &*(kernel->begin()->begin());

  AddrSpaceCastInst *gvp = new AddrSpaceCastInst(gv, gv->getType()->getPointerTo(), "", insertPoint);
  val = new LoadInst(gvp, "", insertPoint);
}
#endif
} // namespace mekong

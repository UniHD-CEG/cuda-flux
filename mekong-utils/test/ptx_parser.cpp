#include <iostream>
#include <string>
#include <cstring>

#include "../inc/ptx_parser.h"

using namespace std;


int main(int argc, char *argv[]) {
  if(argc <= 1) {
    cerr << "Usage:\n\t" << argv[0] << " <filename>\n";
    return -1;
  }
  string filename(argv[1]);

  auto tokenVec = mekong::lexicalAnalysis(filename);
  auto funcVec = mekong::parse(tokenVec);

  for( auto func : funcVec) {
    cout << func.name << '\n';
    for(auto bb : func.bb) {
      cout << "// %" << bb.name << '\n';
      for(auto inst : bb.inst) {
        cout << "\t" << inst << '\n';
      }
    }
  }


  return 0;
}

#include <iostream>
#include <fstream>

#include "../inc/x65599.h"

using namespace std;

int main(int argc, char *argv[]) {
  if(argc <= 1) {
    cerr << "Usage:\n\t" << argv[0] << " <filename>\n";
    return -1;
  }

  const string filename(argv[1]);

  cout << "Hashing file " << filename << "...\n";
  
  ifstream file(filename, std::ios::binary | std::ios::ate);
  auto size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, '\0');
  if( !file.read(&buffer[0], size)) {
    cerr << "Could not read file!\n";
    return -1;
  }

  auto hash = generateHash(buffer.c_str(), size);

  cout << "Hash: " << std::hex << hash << "\n";  

  return 0;
}

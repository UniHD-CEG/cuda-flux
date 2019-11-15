CXX := clang++
RE2C := ../tools/re2c/re2c/re2c

ptx_parser: ptx_parser.cpp ../inc/ptx_parser.h
	$(CXX) ptx_parser.cpp -std=c++11 -o ptx_parser

../inc/ptx_parser.h: ../inc/ptx_parser.hre
	$(RE2C) ../inc/ptx_parser.hre -o ../inc/ptx_parser.h

.PHONY : clean
clean:
	rm ../inc/ptx_parser.h ptx_parser

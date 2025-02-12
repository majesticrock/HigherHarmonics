#include <mrock/utility/InputFileReader.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 2) {
		std::cerr << "Invalid number of arguments: Use mpirun -n <threads> <path_to_executable> <configfile>" << std::endl;
		return -1;
	}

    mrock::utility::InputFileReader input(argv[1]);

    return 0;
}
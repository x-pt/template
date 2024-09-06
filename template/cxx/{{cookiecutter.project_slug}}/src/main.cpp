#include <iostream>
#include "hi.h"
#include "my_lib.h"

int main()
{
    say_hi();
    std::cout << "add(1, 2) = " << add(1, 2) << std::endl;
    std::cout << "sub(1, 2) = " << sub(1, 2) << std::endl;
    return 0;
}

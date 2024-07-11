# include <iostream>

// This will allow us to write cout, cin, endl, etc. instead of std::cout, std::cin, std::endl respectively.
using namespace std;

int main() {
    int number;

    cout << "Enter an integer: ";
    cin >> number;

    cout << "You entered " << number;
    return 0;
}
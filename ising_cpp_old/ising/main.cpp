#include <iostream>
#include <fstream>
#include "metropolis.hpp"

int main()
{
    double epsilon = 0.5;
    int period = 50;
    double T = 1.0;
    //int n_steps = 6000000;
    int L = 100;
    int** lattice = spin_lattice(L);
    
    int** thermalized_lattice = 0;
    //int* E_list = 0;
    std::vector<int> E_list2;
    //std::tie(thermalized_lattice, E_list) = metropolis(lattice, L, T, n_steps);
    std::tie(thermalized_lattice, E_list2) = metropolis2(lattice, L, T, epsilon, period);
    
    //for (int i = 0; i < L; i++) for (int j = 0; j < L; j++) std::cout << thermalized_lattice[i][j] << std::endl;
    
    /*
    std::ofstream myfile ("example.txt");
    if (myfile.is_open())
    {
        myfile << "This is a line.\n";
        myfile << "This is another line.\n";
        myfile.close();
    }
    else std::cout << "Unable to open file";
     */
    
    return 0;
}


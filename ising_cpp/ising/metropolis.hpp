#ifndef metropolis_hpp
#define metropolis_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <tuple>
#include <random>
#include <vector>

/**
 Generates a L*L lattice of Ising spins.
 */
int** spin_lattice(int L);

/**
 Computes the energy of a lattice configuration.
 L: size of the lattice
 */
int energy(int** lattice, int L);

/**
 Implements one step of the Metropolis algorithm.
 L: size of the lattice
 E: energy of the lattice
 T: temperature
 */
int metropolis_one_step(int** lattice, int L, int E, int T);

/**
 Implements n_steps steps of the Metropolis algorithm.
 Keeps track of the energy evolution.
 */
std::tuple<int**, int*> metropolis(int** lattice, int L, int T, int n_steps);

/**
 Variant of the function metropolis, which stops by itself when the running average stops varying too much
 */
std::tuple<int**, std::vector<int>> metropolis2(int** lattice, int L, int T, double epsilon, int period);


#endif /* metropolis_hpp */

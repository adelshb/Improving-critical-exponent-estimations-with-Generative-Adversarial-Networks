#include "metropolis.hpp"

int** spin_lattice(int L)
{
    srand(static_cast<unsigned int>(time(NULL)));
    
    int** lattice = 0;
    lattice = new int*[L];
    
    for (int i=0; i<L; i++)
    {
        lattice[i] = new int[L];
        for (int j=0; j<L; j++)
        {
            lattice[i][j] = 2 * (rand() % 2) - 1;
        }
    }
    return lattice;
}

int energy(int** lattice, int L)
{
    int E = 0;
    for (int i=0; i<L; i++)
    {
        for (int j=0; j<L; j++)
        {
            E = E - lattice[i][j] * (lattice[(i+1+L)%L][j] + lattice[i][(j+1+L)%L]);
        }
    }
    return E;
}

int metropolis_one_step(int** lattice, int L, int E, int T)
{
    int e = E;
    
    int i = rand() % L;
    int j = rand() % L;
    
    int total = 0;
    int nbr[4][2] = {{(i-1+L) % L, j}, {(i+1+L) % L, j}, {i, (j-1+L) % L}, {i, (j+1+L) % L}};
    for (int l=0; l<4; l++) total += lattice[nbr[l][0]][nbr[l][1]];
    
    int dE = 2 * lattice[i][j] * total;
    
    double unif = rand() / (RAND_MAX + 1.);
    if (dE <= 0 || exp(-dE / T) > unif)
    {
        lattice[i][j] *= -1;
        e += dE;
    }
    
    return e;
}

std::tuple<int**, int*> metropolis(int** lattice, int L, int T, int n_steps)
{
    int* E_list = 0;
    E_list = new int[n_steps];
    int e = energy(lattice, L);
    
    for (int i=0; i<n_steps; i++)
    {
        E_list[i] = metropolis_one_step(lattice, L, e, T);
    }
    
    return std::make_tuple(lattice, E_list);
}

std::tuple<int**, std::vector<int>> metropolis2(int** lattice, int L, int T, double epsilon, int period)
{
    std::vector<int> E_list;
    std::vector<double> E_ma;
    int e = energy(lattice, L);
    double sum = 0;
    int count = 0;
    double delta_E = 1.0;
    
    while (delta_E > epsilon)
    {
        E_list.push_back(e);
        sum += e;
        if (count >= period)
        {
            E_ma.push_back((double)sum / period);
            sum -= E_list[count - period];
        }
        e = metropolis_one_step(lattice, L, e, T);
        if (count > period) delta_E = abs(E_ma[count] - E_ma[count - 1]) / E_ma[count - 1];
        count++;
    }
    
    std::cout << "Number of iterations: " << count << std::endl;
    
    return std::make_tuple(lattice, E_list);
}

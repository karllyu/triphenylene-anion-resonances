import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import pi

BASE_DIR = Path(__file__).resolve().parent


def resolve_path(filename):
    path = Path(filename)
    if path.is_absolute():
        return path
    candidate = BASE_DIR / path
    return candidate if candidate.exists() else path


def read_csv(filename):
    col1 = []
    col2 = []

    with open(resolve_path(filename), newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        next(reader, None)

        for row in reader:
            # Skip empty or partial rows (e.g., trailing "," lines in CSVs).
            if len(row) < 2:
                continue
            if row[0].strip() == "" or row[1].strip() == "":
                continue
            col1.append(float(row[0]))
            col2.append(float(row[1]))


    return col1, col2
 

def schlessinger(alphas, energies):
    N = len(alphas)
    coeffs = []

    z0 = 1 / (alphas[1] - alphas[0]) * (energies[0] / energies[1] - 1)
    coeffs.append(z0)

    for i in range(1, N-1):
        numer = []

        for c in range(len(coeffs)):

            num = coeffs[c] * (alphas[i+1] - alphas[c])
            numer.append(num)

        rec = numer[0] / (1 - energies[0] / energies[i+1])

        if len(numer) > 1:
            
            for num in numer[1:]:
                rec = num / (1 + rec)

        denom = 1 / (alphas[i] - alphas[i+1])

        z = denom * (1 + rec)
        coeffs.append(z)

    return coeffs



def continued_fn(alphas, energies, coeffs, eta):
    denominator = 1 + coeffs[-1] * (eta - alphas[-1])

    for val in range(len(alphas) - 2, -1, -1):
        denominator = 1 + coeffs[val] * (eta - alphas[val]) / denominator

    return energies[0] / denominator


    
def complex_etas(alphas, theta_start=0, theta_end=(pi/2), grid_size=91):
    thetas = np.linspace(theta_start, theta_end, grid_size)
    etas = []

    for alpha in alphas:
        for theta in thetas:
            etas.append(alpha * np.exp(1j * theta))

    return etas


def real_continuation(alphas, energies, grid_size=1000):
    coeffs = schlessinger(alphas, energies)
    start = alphas[0] - 1
    end = alphas[-1] + 1
    etas = np.linspace(start, end, grid_size)

    extrapolated_energies = []

    for eta in etas:
        energy = continued_fn(alphas, energies, coeffs, eta)
        extrapolated_energies.append(energy)

    return etas, extrapolated_energies
        

def complex_continuation(alphas, energies, etas):
    coeffs = schlessinger(alphas, energies)

    extrapolated_energies = []

    for eta in etas:
        energy = continued_fn(alphas, energies, coeffs, eta)
        extrapolated_energies.append(energy)

    return etas, extrapolated_energies


def plot_real(stab_plot, points):
    df = pd.read_csv(resolve_path(stab_plot))
    alpha_col = df.columns[0]

    plt.figure()
    for col in df.columns[1:]:
        plt.plot(df[alpha_col], df[col], label=f'{col}')
    
    alphas, energies = read_csv(points)
    plt.scatter(alphas, energies)

    alpha_grid, extrapolated_energies = real_continuation(alphas, energies)
    plt.plot(alpha_grid, extrapolated_energies, label = 'Real Continuation')

    # plt.ylim(0, 3)
    plt.xlabel('Alpha')
    plt.ylabel('Energy')
    plt.legend()
    plt.title('Real Continuation')
    plt.show()


def plot_complex(points, theta_start=0, theta_end=pi, grid_size=91):

    alphas, energies = read_csv(points)
    alphas = np.asarray(alphas, dtype=float)
    energies = np.asarray(energies, dtype=complex)

    coeffs = schlessinger(alphas, energies)

    thetas = np.linspace(theta_start, theta_end, grid_size)

    for alpha0 in alphas:
        etas = alpha0 * np.exp(1j * thetas)

        cont_vals = [continued_fn(alphas, energies, coeffs, eta) for eta in etas]
        cont_vals = np.asarray(cont_vals)

        plt.figure()
        plt.plot(cont_vals.real, cont_vals.imag, marker='o', ms=2, lw=1) 
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.title(f"Theta trajectory at alpha = {alpha0:g}")
        plt.grid(True)

    plt.show()


def plot_complex_fitted(points, stab_points, theta_start=0, theta_end=pi, grid_size=91):

    alphas, energies = read_csv(points)
    alphas = np.asarray(alphas, dtype=float)
    energies = np.asarray(energies, dtype=complex)

    coeffs = schlessinger(alphas, energies)

    thetas = np.linspace(theta_start, theta_end, grid_size)

    etas, energies_cont = read_csv(stab_points)

    for alpha0 in etas:
        etas = alpha0 * np.exp(1j * thetas)

        cont_vals = [continued_fn(alphas, energies_cont, coeffs, eta) for eta in etas]
        cont_vals = np.asarray(cont_vals)

        plt.figure()
        plt.plot(cont_vals.real, cont_vals.imag, marker='o', ms=2, lw=1) 
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.title(f"Theta trajectory at alpha = {alpha0:g}")
        plt.grid(True)

    plt.show()


def test_real(filename):

    alphas, energies = read_csv(filename)

    coeffs = schlessinger(alphas, energies)
    continued = []
    for alpha in alphas:
        continued.append(continued_fn(alphas, energies, coeffs, alpha))

    print('Alphas:')
    print(alphas)
    print('Extrapolated:')
    print(continued)
    print('Data:')
    print(energies)


if __name__ == "__main__":
    # Usage examples:
    plot_real('A2_roots_1-6.csv', 'A2_res1_alt.csv')  # plots real continuation with overlaid points

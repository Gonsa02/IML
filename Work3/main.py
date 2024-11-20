import argparse

from optics.run_optics import run_optics, optics_sort_csv
from spectral.run_spectral import run_spectral, spectral_sort_csv

def main():
    parser = argparse.ArgumentParser(description='Run experiments')
    spectral_sort_csv()

if __name__ == '__main__':
    main()
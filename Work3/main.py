import argparse

from optics.run_optics import run_optics

def main():
    parser = argparse.ArgumentParser(description='Run experiments')
    run_optics()

if __name__ == '__main__':
    main()
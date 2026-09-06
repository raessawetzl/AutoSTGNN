from random_search import run_random_search
from bo_de import run_bode

def bo_de():
      run_bode(
        dataset_name= 'pemsbay',
        model_name='graphwavenet'
      )

def main():
    bo_de()
if __name__ == "__main__":
        main()
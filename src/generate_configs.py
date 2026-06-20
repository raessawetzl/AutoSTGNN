from search_space import get_shared_config_space

if __name__ == "__main__":
    cs = get_shared_config_space()

    print("Hyperparameters defined:")
    for hp in list(cs.values()):
        print(f"  {hp}")

    print("\n5 random sample configs:")
    configs = cs.sample_configuration(5)
    for i, cfg in enumerate(configs):
        print(f"\n  Config {i+1}:")
        for k, v in sorted(dict(cfg).items()):
            print(f"    {k}: {v}")
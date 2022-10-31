import prediction_model_kat as kat
import prediction_model_sunmin as sunmin
import prediction_model_vincent as vincent
import prediction_model_mig as mig


def run_models():
    sunmin.run()
    kat.run()
    vincent.run()
    # mig.run()


if __name__ == "__main__":
    run_models()

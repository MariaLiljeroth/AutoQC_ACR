from pathlib import Path
import pickle
import logging
from platformdirs import user_cache_dir


class CacheManager:

    def __init__(self, app_name: str):
        self.cache_dir = Path(user_cache_dir(app_name))
        self.cache_dir.mkdir(exist_ok=True)

    def store_cache(self, data, cache_name: str):
        cache_file = self.cache_dir.joinpath(cache_name)
        try:
            with open(cache_file, "wb") as file:
                pickle.dump(data, file)
                logging.info(f"Data successfully stored to cache: {cache_file}")

        except Exception as e:
            logging.exception(f"Error storing to cache: {e}")
            raise

    def load_cache(self, cache_name: str):
        cache_file = self.cache_dir.joinpath(cache_name)
        try:
            with open(cache_file, "rb") as file:
                data = pickle.load(file)
                logging.info(f"Pulling data from cache from cache name '{cache_name}'!")
                return data

        except (pickle.UnpicklingError, EOFError) as e:
            logging.exception(f"Error loading cache: {e}")
            raise

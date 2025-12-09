import configparser
import os

class Config:
    """
    Manages global configuration settings loaded from an external .ini file.
    
    This class handles finding the configuration file relative to the source directory
    and parsing settings for data paths, model preferences, and execution parameters.
    """
    def __init__(self):
        """
        Initializes the configuration by locating and reading 'config.ini'.

        It attempts to locate 'config/config.ini' relative to the project structure.
        If the file is not found, it uses default values instead of raising an error.
        """
        self.config = configparser.ConfigParser()

        # Take the path of file src/modeltrainer/config.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # current_dir = .../PROJECT/src/modeltrainer

        # Try multiple paths to find config.ini
        config_paths = [
            os.path.join(current_dir, '..', '..', 'config', 'config.ini'),  # PROJECT/config/config.ini
            os.path.join(current_dir, '..', 'config', 'config.ini'),  # PROJECT/src/config/config.ini (fallback)
            'config.ini',  # Current working directory
            os.path.join(current_dir, '..', '..', 'config.ini'),  # PROJECT/config.ini
        ]
        
        config_path = None
        for path in config_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                config_path = normalized_path
                break
        
        if config_path:
            print(f"Reading config from: {config_path}")
            self.config.read(config_path)

            # Load DATA SECTION
            self.DATA_PATH = self.config.get('DATA', 'data_path')
            self.TEST_SIZE = self.config.getfloat('DATA', 'test_size')
            self.RANDOM_STATE = self.config.getint('DATA','random_state')

            # Load MODEL SECTION
            self.DEFAULT_MODEL = self.config.get('MODEL','default_model')
            self.N_TRIALS = self.config.getint('MODEL','n_trials')
        else:
            # Config file not found, use default values
            print("Warning: Config file not found. Using default values.")
            self._set_defaults()

    def _set_defaults(self):
        """Sets default configuration values when config file is not found."""
        self.DATA_PATH = 'data/vietnam_housing_dataset.csv'
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        self.DEFAULT_MODEL = 'RandomForestRegressor'
        self.N_TRIALS = 50

    def update_from_args(self,args):
        """
        Updates configuration attributes based on command-line arguments.

        This method allows overriding the default values loaded from the .ini file
        with runtime arguments provided by the user via argparse.

        Parameters
        ---
        args : argparse.Namespace
            The parsed command-line arguments containing potential overrides (e.g., data_path, model, trials).
        """
        if args.data_path:
            self.DATA_PATH = args.data_path
        if args.model:
            self.DEFAULT_MODEL = args.model
        if args.trials:
            self.N_TRIALS = args.trials
        if args.test_size is not None:
            self.TEST_SIZE = args.test_size
        if args.random_state is not None:
            self.RANDOM_STATE = args.random_state

# Initialize a default instance - always succeeds with default values if file not found
# Since __init__ now handles missing config files gracefully, settings will never be None
try:
    settings = Config()
except Exception as e:
    # Safety net: if something unexpected happens, create instance with defaults
    print(f"Warning: Unexpected error initializing config: {e}. Using default values.")
    settings = Config()
    # Force defaults in case __init__ partially succeeded
    settings._set_defaults()
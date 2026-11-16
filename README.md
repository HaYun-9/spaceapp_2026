# spaceapp_2026
Submission for nasa space apps 2026 challenge

# INSTALLATION AND USAGE
- Python 3.11.14 was used to develop and run the application.
- pkg_list.txt shows all the libraries used to run the local conda environment.
- in additon, pytorch 2.7.1 + cuda support were utilized for the machine learning aspect.

# USAGE
- Using anaconda, run the command `conda create --name <env> --file pkg_list.txt` to initialize an environment with all the packages.
    - Additionally, pytorch 2.7.1 is required to be installed to run the created machine learning model.
- Run the python file application.py: `python3 application.py`
- The program will use the geocoder package to obtain the user's current longitude/latitude
- Once the script has fetched their current climate data and run the data through the machine learning model, the script terminates by listing which of all the observed crop/environment types is likely to occur in the given location (and thus what is likely to perform well), ranked highest to lowest confidence in success.

# TROUBLESHOOTING
- The program will only work in a longitude/latitude bounding box over California.
- It is unknown if newer Pytorch versions will allow this project to run.

# NOTES AND DISCLAIMERS
- NASA POWER and the Cropland Data Layers offered by CropScape and the USDA were the datasets used to build this project.
- Due strictly to time constraints, Gemini was used to help with code optimization (specifically during the development of the machine learning model) and debugging.
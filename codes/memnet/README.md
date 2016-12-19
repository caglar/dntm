
First run,

python run_models.py --exp <proto_name>

The script above will create the DB tables and add the experiments into the DB.

Then launch the experiments with ./jobdispatch.sh:

./jobdispatch.sh <table_name> 20

# mlim-g2
Group 2: Final submission, Machine Learning in Marketing @HU Berlin winter term 20-21

## Organization
__Authors:__ Sascha Geyer, Leon Hagel, Anna Langenberg, Benedikt Rein, Jessica Ströbel <br>
__Lecture:__ Machine Learning in Marketing <br>
__Institute:__ Humboldt University Berlin, Institute of Marketing <br>
__Lecturer:__ Dr. Sebastian Gabel <br>
__Semester:__ WS 2020/21 <br>

## Content 

```
.
├── cache                     # cached files created while running the pipelin
├── input                     # input to out pipeline (i.e. data provided via moodle)
├── output						# output created by our pipeline (i.e. final output, images)
├── src                       # necessary code to run the pipeline
├── config.json               # configuration for running the pipeline
├── Makefile                  # run `make help` to see make targets
├── README.md                 # this readme file
├── requirements.txt          # virtualenv requirements file
└── task.pdf                  # task description for the final project
```


## Requirements

1. This project is implemented with Python 3.8.
1. `virtualenv`
1. Input data provided via moodle, i.e. `baskets.parquet`, `coupons.parquet`.


## Running our coupon optimization

1. Clone the mlim-g2 repository from github.
1. Copy the input data provided via moodle to `./data`.
1. Run `make build` or `make build-lab` to create the virtual environment located at `./.env`.
1. Run `make create` to create our final output containing the optimized coupons.

## Makefile targets

```
build          install dependencies and prepare environment located at ./.env
build-lab      build + lab extensions
freeze         view installed packages
clean-cache    remove all files in the cache directory
clean          clean-cache + remove *.pyc files and __pycache__ directory
distclean      clean +  remove virtual environment
lab            run jupyter lab (default port 8888)
create-plots   create plots for the report
create         create final output containing the optimized coupons
```

## References

This readme and the makefile are based on the files used by Dr. Sebastian Gabel in the GitHub repository for the course ([sbstn-gbl/mlim](https://github.com/sbstn-gbl/mlim "Repository for Course Machine Learning in Marketing")).
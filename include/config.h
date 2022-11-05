#ifndef _config_h
#define _config_h

#define FILE_PATH "./data/dataset.csv"
#define FILE_PATH_AUGMENTED "./data/augmented_dataset.csv"

#define NUM_RUNS        1

#define MAX_FREQ        3.2
#define BASE_FREQ       2.4

#define AUGMENT_FACTOR  5

#define OBSERVATIONS	112
#define FEATURES	    16

#define UNDEFINED        0
#define CLASS_1          1
#define CLASS_2          2
#define CLASS_3          3
#define CLASS_4          4
#define CLASS_5          5
#define CLASS_6          6
#define CLASS_7          7
#define NOISE            8

#define EPSILON          1.70
#define MINPTS           4

// datatype of feature space
#define DTYPE float

// datatype of observations iterator
#define DTYPE_OBS unsigned long long
#endif

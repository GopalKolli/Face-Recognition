							README

Face Datasets: YaleB Face dataset

I discourage any student to use this repository for academic projects/assignments and other academic purposes.

This project consists of two MATLAB scripts.

'YaleB_D-KSVD.m' is to read all the images in YaleB database and prepare the image_vector_matrix and H_Label_matrix. This script then trains the D-KSVD model with 1600 samples and tests with 800 samples and gives out accuracy result.

'Synthetic_Data_DKSVD.m' is to generate synthetic vectors and prepare the vector_matrix and H_Label_matrix. This script then trains the D-KSVD model with 600 synthetic samples and tests with 190 synthetic samples and gives out accuracy result.

At the beginning of each MATLAB script, regulatory parameters like number_of_iterations and number_of _atoms _in_dictionary can be modified before running the MATLAB script.

The algorithm may take around 15 minutes to complete execution. Kindly wait for the results.




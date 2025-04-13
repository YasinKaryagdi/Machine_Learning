## Machine Learning Project
I wrote the initial code for the project after which I finalized the code together with my group. This repo was used in order to share the code for training, validating and testing the classifiers. The actual final runnable code can be found in the appendix of the final report.

**_NOTE:_** These files were written using Jupyter Notebook, I'm somewhat confused on how they are currently .py files instead of .ipynb files. I couldn't find a direct solution without radically changing the repo and its history so I decided to keep the files as .py files. Since the files are still runnable I saw this as a minor issue. Furthermore there were other files used for analyzing and preprocessing but which are now lost permanently, though the working lines of code of these lost files have been used in the final code found in the appendix.

# datasetsnew.py
The dataset contains 24 million instances of data which caused us to run into some issues with running and sharing the code. Specifically in order to upload the dataset on GitHub the files were required to be below a certain size. In order to solve this issue the original csv files were split into 50 smaller files.

# MergingData.py
Code written in order to test the functionality of splitting and merging back the data, making sure that everything was in order.

# FinalTest.py
This was the general code before we finalized it to solve some issues we ran into with this code. This code reads the csv files, partitions the dataset, trains and tests the models after which it generates the results and their graphical representations. The main issue we ran into with this code is that there is too much data, which causes the kernel to crash. We tried to circumvent this issue by running it on a university research computer. But that approach didn't end up working since we failed to set up a valid virtual environment in time, thus we decided to narrow down the scope of the data used for our final test.

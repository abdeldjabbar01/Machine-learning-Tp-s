# Work Session 04: Overfitting and Data Leakage

* In this project, I used a student survey to predict final CGPA.
* I worked through the notebook and learned how to test models more carefully.

## What I tried

* I compared a few models and used data from an earlier period for training and a later period for testing. 
* I also looked at how duplicates and information from the target can lead to data leakage.

## What I learned

- A model can do very well on training data but poorly on new data.
- Keeping the test data separate helps me get a more honest result.
- A simple model can work as well as a more complex one. Linear Regression and Gradient Boosting had similar test scores.
- The results were modest, which makes sense because the survey has limited information about student performance.

The notebook has the code and full results. Next, I’d like to try predicting CGPA bands instead of predicting a number.

## Files

- `Tp-04_completed.ipynb` — notebook
- `hybrid_student_performance_1200.csv` — dataset
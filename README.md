<!-- PROJECT Description -->

<br>
<p align="center">

<<<<<<< HEAD
<h3 align="center">A predictive analysis application for Feature Optimization using Genetic Algorithm</h3>
=======
  <h3 align="center">A predictive Analysis System for Feature Optimization using Genetic Algorithm</h3>
>>>>>>> 1aca69a6de52b10cd7cc2fb60336a9d48f75b615
<br>

<!-- Installation Instruction -->

# How to run code

## 1. Clone the repository

```sh
https://github.com/hbkabir004/Feature-Optimization-using-Genetic-Algorithm.git
```

## 2. Install Python

2.1 Download the Python Installer binaries.

```sh
https://www.python.org/downloads/
```

<br>
2.2  Run the Executable Installer.
<br>
2.3  Add Python to environmental variables.
<br>
2.4  Verify the Python Installation.
<br>

## 3. Install & Import Necessary Python Packages in Visual Studio Code

Follow the tutorial to install & import necessary python packages in VS Code

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/paRXeLurjE4/default.jpg)](https://youtu.be/paRXeLurjE4)
`<br>`

## 4. Open the project in the IDE (VS CODE recommended)

Follow the tutorial to Run Python in Visual Studio Code on Windows 10

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/AKVRkB0fot0/default.jpg)](https://youtu.be/AKVRkB0fot0)

## 5. RESULTs

### 5.1 RANDOM FOREST

##### Optimal Feature Set

```sh
['radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'radius_se', 'perimeter_se', 'concavity_se', 'radius_worst', 'texture_worst', 'area_worst', 'concave points_worst', 'fractal_dimension_worst']
```

##### Feature Importances

```sh
[0.0345544  0.01528677 0.05257772 0.04909357 0.00796349 0.0073864
 0.03402101 0.0742245  0.00459763 0.00389691 0.00996118 0.00515219
 0.01823021 0.04250951 0.00312534 0.00529447 0.00446035 0.00320924
 0.00378835 0.00521066 0.14585278 0.02144762 0.14566475 0.09814828
 0.01441097 0.01654489 0.04011699 0.1168934  0.01104655 0.00532988]
```

```sh
Optimal Accuracy = 99 %

Average Accuracy saved 0.961335676625659
Average Precision      0.9612987777153709
Average Recall         0.9557766502827546
Average F1-Score       0.9584064327485381

     B    M
B  349    8
M   14  198

```

### 5.2 Light GBM

##### Optimal Feature Set

```sh
 ['smoothness_mean', 'concavity_mean', 'concave points_mean', 'perimeter_se', 'area_se', 'concave points_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
```

##### Feature Importances

```sh
[ 69 170  49  36  65  64  88 174  81  54  80  79  48 142  54  43  27  47
  57  47  92 282 147 144 104  40 109 206 107  42]
```

```sh
Optimal Accuracy = 99 %

Average Accuracy saved 0.9718804920913884
Average Precision 0.9707985143918292
Average Recall 0.9689696633370328
Average F1-Score 0.9698694696708942

     B      M
B   350     7
M   9     203
```

### 5.3 XGBoost

##### Optimal Feature Set

```sh
['perimeter_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'fractal_dimension_mean', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'texture_worst', 'area_worst', 'compactness_worst', 'concave points_worst', 'symmetry_worst']
```

##### Feature Importances

```sh
[0.00523759 0.01394322 0. 0.01937084 0.00388636 0.00419483
0.00792941 0.12601759 0.00246526 0.00336443 0.00851927 0.01265129
0.00763103 0.00893879 0.00833281 0.0060065 0.01219247 0.01192982
0.00208353 0.00228187 0.3791942 0.0181876 0.19956343 0.01817427
0.00790597 0.00302455 0.01822192 0.07777867 0.00269732 0.0082751 ]
```

```sh
Optimal Accuracy = 99 %

Average Accuracy saved 0.9648506151142355
Average Precision 0.966116035455278
Average Recall 0.9585777707309339
Average F1-Score 0.9621111229490731

     B      M
B   351     6
M   14    198
```

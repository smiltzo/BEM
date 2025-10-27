# uBEM

A **Python-based Blade Element Momentum (BEM) model** for analysing wind turbine performance. Based on the DTU 10MW reference wind turbine[1].
This project aims to develop a modular and transparent framework for aerodynamic performance prediction and wake modelling.

---

## Overview
uBEM (unsteady Blade Element Momentum) is a lightweight and flexible simulation tool that computes aerodynamic loads and rotor performance using BEM theory.  
It is designed for research and educational purposes, enabling easy modification and extension.

---

## Features
- Sheared wind model
- Tower effect (Potential flow theory)
- Dynamic Inflow model
- Dynamic Stall (S. Øye model)
- Yaw model
- Turbulent inflow (Mann model)

---

## Technologies Used
- **Python 3.11**
- **NumPy**, **Pandas**, **Matplotlib**
- **SciPy** for interpolation
- **Dataclasses** to store quantities

---

## Folder Structure
- **Project Folder**
   - *src* contains the code files:
      - **BEM.py** -> main file
      - **BEM_dataclasses.py** -> contains datalcasses
      - **BEM_utils.py** -> contains functions
   - *Data* contains .csv data files:
      - **bladedat.csv** -> contains blade structural data and radius span
      - The other files contain aerodynamic polars – Cl, Cd, Cm, Cl inviscid, Cl fully separated and separation factor (fs)

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/smiltzo/uBEM.git

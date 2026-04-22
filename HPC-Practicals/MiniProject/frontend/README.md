# 🚀 HPC-Based Parallel Machine Learning Training System

## 📌 Project Description

This project demonstrates the use of **High Performance Computing (HPC)** techniques to speed up Machine Learning model training.

We compare four different training approaches:

- 🧠 Serial Training (single core)
- ⚡ Parallel Training (multi-core using `n_jobs`)
- 🔥 Advanced Parallel Training (manual chunking using Joblib)
- 🚀 MPI Training (distributed computing using multiple processes)

The system includes:
- 🖥️ Backend (Node.js + Python ML scripts)
- 🎨 Frontend (React Dashboard)
- 📊 Visualization (Chart.js graph comparison)

---

## 🎯 Objective

- Reduce ML training time using parallel computing  
- Demonstrate scalability of HPC systems  
- Compare performance of different execution strategies  
- Visualize results using a web dashboard  

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|-----------|
| Frontend | React.js |
| Backend | Node.js + Express |
| ML | Python (Scikit-learn) |
| HPC | Joblib, MPI (mpi4py) |
| Visualization | Chart.js |

---

## 📁 Project Structure

```
MiniProject/
│
├── backend/
│   ├── ml/
│   │   ├── data/
│   │   │   └── crop_data_large.csv
│   │   ├── serial_train.py
│   │   ├── parallel_train.py
│   │   ├── advanced_parallel.py
│   │   └── mpi_train.py
│   └── server.js
│
├── frontend/
│   ├── src/
│   │   ├── pages/TrainingPage.jsx
│   │   └── App.jsx
```

---

## ⚙️ Installation Guide

### 🔹 1. Install Python Dependencies

```bash
pip install pandas scikit-learn mpi4py joblib
```

---

### 🔹 2. Install MPI (Windows)

- Install MS-MPI Runtime  
- Install MS-MPI SDK  
- Restart your system  

Verify installation:
```bash
mpiexec
```

---

### 🔹 3. Install Backend Dependencies

```bash
cd backend
npm install express cors
```

---

### 🔹 4. Install Frontend Dependencies

```bash
cd frontend
npm install
```

---

## ▶️ How to Run the Project

### ✅ Step 1: Start Backend Server
```bash
cd backend
nodemon server.js
```

Expected output:
```
🚀 Server running on port 5000
```

---

### ✅ Step 2: Start Frontend
```bash
cd frontend
npm run dev
```

Open in browser:
```
http://localhost:fronend_port
```

---

### ✅ Step 3: Run Training from UI

Click the buttons:
- Run Serial  
- Run Parallel  
- Run Advanced  
- Run MPI  

---

## ⚡ Run MPI Manually (Optional)

```bash
cd backend/ml
mpiexec -n 4 python mpi_train.py
```

---

## 📊 Output

The dashboard displays:
- Training time for each method  
- Graph comparison  

| Method   | Time  |
|----------|-------|
| Serial   | 5.13s |
| Parallel | 1.20s |
| Advanced | 3.95s |
| MPI      | 2.81s |

---

## 🧠 Key Concepts Used

- Parallel Computing  
- Multi-core Processing  
- Distributed Computing (MPI)  
- Machine Learning Optimization  
- Performance Analysis  

---

## 🎯 Advantages

- Faster ML training  
- Efficient CPU utilization  
- Scalable architecture  
- Real-world HPC simulation  

---

## ⚠️ Notes

Ensure dataset path is correct:
```
ml/data/crop_data_large.csv
```

If MPI command fails, use full path:
```bash
"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
```

---

## 🚀 Future Enhancements

- GPU-based training (CUDA)  
- Cloud deployment (AWS / Azure)  
- Real-time dataset scaling  
- Advanced analytics dashboard  


## 🏁 Conclusion

This project demonstrates how HPC techniques significantly reduce ML training time.

It combines:
- ✔ Machine Learning  
- ✔ Parallel Computing  
- ✔ Distributed Systems  
- ✔ Full-stack Development  

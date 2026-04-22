import React, { useState } from "react";
import axios from "axios";

import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

function TrainingPage() {
  const [serial, setSerial] = useState("");
  const [parallel, setParallel] = useState("");
  const [advanced, setAdvanced] = useState("");
  const [mpi, setMpi] = useState("");
  const [loading, setLoading] = useState(false);

  const runTraining = async (type) => {
    try {
      setLoading(true);

      const res = await axios.get(`http://localhost:5000/train/${type}`);

      if (type === "serial") setSerial(res.data);
      if (type === "parallel") setParallel(res.data);
      if (type === "advanced") setAdvanced(res.data);
      if (type === "mpi") setMpi(res.data);

    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const extractTime = (text) => {
    if (!text || typeof text !== "string") return 0;

    const timeMatch = text.match(/(?:Training Time|Average Time|Time):\s*([\d.]+)/i);
    if (timeMatch) {
      const val = parseFloat(timeMatch[1]);
      return isNaN(val) ? 0 : val;
    }

    const match = text.match(/[\d.]+/);
    if (match) {
      const val = parseFloat(match[0]);
      return isNaN(val) ? 0 : val;
    }

    return 0;
  };

  const serialTime = extractTime(serial);
  const parallelTime = extractTime(parallel);
  const advancedTime = extractTime(advanced);
  const mpiTime = extractTime(mpi);

  const data = {
    labels: ["Serial", "Parallel", "Advanced", "MPI"],
    datasets: [
      {
        label: "Training Time (seconds)",
        data: [serialTime, parallelTime, advancedTime, mpiTime],
        backgroundColor: [
          "#ff4d4f",
          "#1890ff",
          "#52c41a",
          "#722ed1",
        ],
        borderRadius: 6,
      },
    ],
  };

  const options = {
    maintainAspectRatio: false,
    responsive: true,
    plugins: {
      legend: {
        display: true,
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            return `${context.raw.toFixed(4)} sec`;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function (value) {
            return value + "s";
          },
        },
      },
    },
  };

  const buttonStyle = {
    padding: "10px 15px",
    margin: "10px",
    borderRadius: "8px",
    border: "none",
    backgroundColor: "#007bff",
    color: "white",
    cursor: "pointer",
  };

  const cardStyle = {
    padding: "15px",
    marginTop: "20px",
    borderRadius: "10px",
    backgroundColor: "#f5f5f5",
  };

  return (
    <div style={{ padding: "20px", minHeight: "100vh", backgroundColor: "#f0f2f5" }}>
      <h2>HPC ML Training Dashboard</h2>

      <div>
        <button style={buttonStyle} onClick={() => runTraining("serial")}>
          Run Serial
        </button>

        <button style={buttonStyle} onClick={() => runTraining("parallel")}>
          Run Parallel
        </button>

        <button style={buttonStyle} onClick={() => runTraining("advanced")}>
          Run Advanced
        </button>

        <button style={buttonStyle} onClick={() => runTraining("mpi")}>
          Run MPI Training
        </button>
      </div>

      {loading && <p>⏳ Training in progress...</p>}

      <div style={cardStyle}>
        <h3>Results</h3>
        <p><b>Serial:</b> {serialTime.toFixed(4)} sec</p>
        <p><b>Parallel:</b> {parallelTime.toFixed(4)} sec</p>
        <p><b>Advanced:</b> {advancedTime.toFixed(4)} sec</p>
        <p><b>MPI:</b> {mpiTime.toFixed(4)} sec</p>
      </div>

      <div style={{ marginTop: "30px", height: "400px", background: "#fff", padding: "20px", borderRadius: "10px" }}>
        <Bar data={data} options={options} />
      </div>
    </div>
  );
}

export default TrainingPage;
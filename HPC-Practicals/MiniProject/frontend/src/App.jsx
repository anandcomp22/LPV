import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import TrainingPage from "./pages/TrainingPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<TrainingPage />} />
        <Route path="/training" element={<TrainingPage />} />
      </Routes>
    </Router>
  );
}

export default App;
const express = require("express");
const { exec } = require("child_process");
const cors = require("cors");

const app = express();
app.use(cors());

app.get("/train/serial", (req, res) => {
  exec("python ml/serial_train.py", (err, stdout, stderr) => {
    if (err) return res.send(stderr);
    res.send(stdout);
  });
});

app.get("/train/parallel", (req, res) => {
  exec("python ml/parallel_train.py", (err, stdout, stderr) => { 
    if (err) return res.send(stderr);
    res.send(stdout);
  });
}); 

app.get("/train/advanced", (req, res) => {
  exec( 
    `"C:\\Program Files\\Microsoft MPI\\Bin\\mpiexec.exe" -n 4 python ml/advanced_parallel.py`,
    (err, stdout, stderr) => {
      if (err) return res.send(stderr);
      res.send(stdout);
    }
  );
});

app.get("/train/mpi", (req, res) => {
  exec(
    `"C:\\Program Files\\Microsoft MPI\\Bin\\mpiexec.exe" -n 4 python ml/mpi_train.py`,
    (err, stdout, stderr) => {
      if (err) return res.send(stderr);
      res.send(stdout);
    }
  );
});

app.listen(5000, () => console.log("Server running on port 5000"));
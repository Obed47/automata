# Quasi-Newton SR1 Optimization

🚀 A multi-language implementation of the **Quasi-Newton Symmetric Rank 1 (SR1)** algorithm for optimization problems, written in **Rust** and **Lua**.

---

## 📘 Overview

The **Quasi-Newton SR1 algorithm** is an iterative method used for unconstrained optimization. It approximates the inverse Hessian matrix using a rank-1 update strategy, allowing faster convergence without computing second derivatives.

This repository contains two implementations:
- 🦀 `rust/` – a fast and safe implementation using **Rust**.
- 🌙 `lua/` – a lightweight and scriptable version using **Lua**.

---

## 📁 Project Structure

├── rust/ # Rust implementation of SR1
│ ├── Cargo.toml
│ └── src/
│ └── main.rs
├── lua/ # Lua implementation of SR1
│ └── sr1.lua
└── README.md 

---

## 🦀 Running the Rust Version

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) installed

### Build and Run

```bash
cd rust
cargo run

```
## Running the lua version
###Prerequisites
install lua with sudo apt install lua5.3

###Build and run
```bash
cd lua
lua sr1.lua

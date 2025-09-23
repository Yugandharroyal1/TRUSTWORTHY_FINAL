Project1-README

Project Title: GradOS Project 1 – Xv6 User Programs and System Calls
Student Name: Yugandhar
UID: U32413262
Group Members: None

1. System Environment

Operating System: Ubuntu 22.04 LTS

Xv6 Version: xv6-public (latest clone from MIT GitHub)

Compiler: GCC version 11.4.0

QEMU Version: qemu-system-i386 6.2.0

Terminal: GNOME Terminal

Make: GNU Make 4.3

2. Project Overview

The project focuses on adding user programs and system calls in Xv6 and modifying existing commands.

Four main tasks:

Hello World Program (hello)

Modified ls Command

Hello World Syscall (hello syscall)

Sleep Command (sleep)

3. Step-by-Step Implementation
Step 0: Ubuntu and Xv6 Setup

Open terminal.

Navigate to project folder:

cd ~/osclass/xv6-public


Explore folder structure to understand Xv6 environment.

Step 1: Cleaning Previous Builds

Remove old compiled files:

make clean

Step 2: Compiling Xv6

Compile all programs including new user programs:

make

Step 3: Running Xv6

Launch Xv6 in GUI mode:

make qemu


Or launch Xv6 in terminal-only mode:

make qemu-nox


Use Xv6 shell to test commands:

$ hello
$ ls
$ ls -s
$ sleep 100

Part 1 – Hello World Program

Objective: Print Hello Xv6! to stdout.

Steps:

Created hello program.

Added program to Makefile under UPROGS.

Verified compilation:

make


Ran program in Xv6 shell:

$ hello

Part 2 – Modified ls Command

Objective: Improve readability.

Changes:

Skip hidden files.

Append / to directories.

Add optional -s flag to sort files by size.

Steps:

Modified ls.c to implement changes.

Verified compilation:

make


Tested in Xv6 shell:

$ ls
$ ls -s

Part 3 – Hello World Syscall

Objective: Add a kernel-level system call.

Steps:

Added syscall function in kernel.

Updated user.h and usys.S.

Called syscall from user program.

Verified in Xv6 shell:

$ hello

Part 4 – Sleep Command

Objective: Pause execution for user-specified ticks.

Steps:

Created sleep program.

Added program to Makefile under UPROGS.

Compiled using:

make


Ran program in Xv6 shell and confirmed behavior:

$ sleep 100

4. Testing & Screenshots

Compiled programs with:

make clean && make


Tested using:

make qemu-nox


Screenshots captured for:

Program code files

Compilation process (make clean, make)

Xv6 shell outputs for all commands

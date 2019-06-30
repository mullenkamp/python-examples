Python examples repo
==================================

This git repository contains python example code for various purposes for ECan scientists and analysts. Some examples may work from outside the ECan network, but no guarantees.

To get started, follow the steps below.

Get a GitHub account
--------------------
Go to `GitHub <https://github.com>`_ and register for an account. Then tell an admin of the `Data-to-Knowledge <https://github.com/Data-to-Knowledge>`_ GitHub organisation your username so that you can join the organisation.

Download and install GitHub Desktop
-----------------------------------
Go to `<https://desktop.github.com/>`_ and download and install GitHub Desktop. No admin rights required. This program helps you manage all of your Git repositories. A user guide can be found here: `<https://help.github.com/en/desktop>`_.

Learn some Git/GitHub basics
----------------------------
Talk to someone who is familiar with Git/GitHub and/or read through the tutorials described on the `GitHub help pages <https://help.github.com/en#dotcom>`_.

Clone this repo and create your own repo
----------------------------------------
Clone this repository to your own PC, create a new repo either under your own user account or under Data-to-Knowledge, and copy over the files and folders from this repo to your new one. Then make your first commit and push to your new repo.

Download and install Miniconda
------------------------------
Download and install the recommended Python installation called `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. No admin rights required. A user guide can be found here: `<https://docs.conda.io/projects/conda/en/latest/user-guide/index.html>`_. A nice "Cheat sheet"  can be downloaded here: `<https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html>`_.
There's a lot you can do with this Python installation. It's definitely worth it to talk to others who are familiar with it's inner workings.

Create a Python environment to run the example code
---------------------------------------------------
The main.yml file defines the required packages dependencies for all of the examples.

To install via conda, type the following into the anaconda prompt when in the root directory::

  conda env create -f main.yml

Run the code editor Spyder to run the Python code
-------------------------------------------------
Now that the code editor Spyder is installed on your PC, you can open Spyder and run the example code via Spyder.

Learning Python
---------------
Follow the courses on the `Ecan Python Courses <https://ecanpythoncourse2019.readthedocs.io/en/latest/>`_ to learn the basics of Python and handling tabular data.

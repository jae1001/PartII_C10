## iPython Notebooks for Part II Materials Science C10: Polymers

<ol>
<li>

[3D random walk: 3D_random walk.ipynb](https://colab.research.google.com/github/jae1001/PartII_C10/blob/main/3D%20random%20walk.ipynb)

**Description**

Simple Monte Carlo cubic lattice polymer chain model to demonstrate scaling behaviour
of a non-self avoiding (ideal) random walk chain. For such a chain, the root-mean-square
end-to-end distance of an "long" chain (N>100) of segments of unit length should be equal
to sqrt(N).

**Usage**

User-changeable parameters are the number of monomer segments in each chain (N), and
number of configurations (M) to average over. Since we are using end-to-end distance, we
expect large fluctations in sqrt(N), which are exacerbated by the use of a cubic lattice,
so do not expect good covergence.

**Credits**

Original Python code written by James Elliott
</li>


<li>

[script.plt](https://colab.research.google.com/github/jae1001/PartII_C10/blob/main/script.plt)

**Description**

Gnuplot script to fit triple-Gaussian (+background) curve to azimuthally integrated X-ray
scattering from semi-crystalline polyethylene sample. Relies on data file "pe_rect.txt" being
present in same directory as script is loaded from.

**Usage**

Unpack raw data from "Ex. Class Q5.zip" archive and put in same directory as script. Load the 
script into Gnuplot ([gnuplot.info](http://gnuplot.info/).

**Credits**

Original Gnuplot script written by James Elliott
</li>

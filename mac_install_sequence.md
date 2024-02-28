conda create -n "coastmac" python=3.8.18
conda config --env --add channels conda-forge
conda install geopandas earthengine-api scikit-image matplotlib astropy notebook -y
conda install pyqt5-sip imageio-ffmpeg pillow opencv diffusers==0.11.1 -y
conda install anaconda::pyqt   
conda install orbingol::geomdl




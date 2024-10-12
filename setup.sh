pip3 install -r requirements.txt
git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
cd sparse_tools
python setup.py develop --user
cd ..

pip3 install pydantic
pip3 install sortednp
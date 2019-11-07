import urllib.request

url = 'https://www1.ncdc.noaa.gov/pub/data/cmb/ersst/v5/netcdf/'

for y in range(1971,2015):
    for m in ["%.2d" % i for i in range(1,13)]:
        urllib.request.urlretrieve(url, 'ersst.v5.{}{}.nc'.format(y,m))

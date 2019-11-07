mkdir ONI

### Merge sincle years and select sea surface temperature from file
cdo mergetime ersst.v5* ersst.v5.1971-2015.nc
cdo selvar,sst ersst.v5.1971-2015.nc ONI/SST_1971-2015.nc

### Select Nino3.4 region
cdo sellonlatbox,-170,-120,5,-5 ONI/SST_1971-2015.nc ONI/SST_1971-2015_nino34.nc

### Calculate base periods updated every five years and create climatology
for start in {1986..2005..5}
    do   
    clim_start=$((start - 15)) 
    clim_end=$((start + 14)) 

    cdo -ymonmean -selyear,$clim_start/$clim_end ONI/SST_1971-2015_nino34.nc \
    ONI/$clim_start'_'$clim_end'.'nc
	
    ind_year1=$start
    ind_year2=$((start+1))
    ind_year3=$((start+2))
    ind_year4=$((start+3))
    ind_year5=$((start+4))
    
    cdo settaxis,$ind_year1'-'01-01,00:00,1month ONI/$clim_start'_'$clim_end'.'nc \
    ONI/$ind_year1'_'clim.nc
    
    cdo settaxis,$ind_year2'-'01-01,00:00,1month ONI/$clim_start'_'$clim_end'.'nc \
    ONI/$ind_year2'_'clim.nc
    
    cdo settaxis,$ind_year3'-'01-01,00:00,1month ONI/$clim_start'_'$clim_end'.'nc \
    ONI/$ind_year3'_'clim.nc
    
    cdo settaxis,$ind_year4'-'01-01,00:00,1month ONI/$clim_start'_'$clim_end'.'nc \
    ONI/$ind_year4'_'clim.nc
    
    cdo settaxis,$ind_year5'-'01-01,00:00,1month ONI/$clim_start'_'$clim_end'.'nc \
    ONI/$ind_year5'_'clim.nc

    rm ONI/$clim_start'_'$clim_end'.'nc
    done

cdo mergetime ONI/*_clim.nc ONI/climatology.nc

### Calculate anomaly
cdo sub ONI/SST_1971-2015_nino34.nc ONI/climatology.nc ONI/SST_anomaly.nc

### Calculate spatial average and 3-month running mean
cdo -runmean,3 -fldmean ONI/SST_anomaly.nc ONI/ONI.nc

### Remove temporary files
rm ONI/*_clim.nc
rm ONI/SST_1971-2015.nc
rm ONI/SST_1971-2015_nino34.nc
rm ONI/climatology.nc
rm ONI/SST_anomaly.nc
rm ersst.v5.1971-2015.nc

def create_mosaic(path, field_name, obs_date):

    import matplotlib.pyplot as plt
    import glob
    import numpy as np
    from astropy.wcs import WCS
    import glob
    import astropy.io.fits as fits
    import os
    from astropy.time import Time
    from datetime import datetime, timedelta

    from reproject import reproject_interp
    from reproject.mosaicking import reproject_and_coadd
    from reproject.mosaicking import find_optimal_celestial_wcs

    middlelink = '/obs/lenses_EPFL/PRERED/VST/reduced/' + field_name + '/'
    allothers = '/obs/lenses_EPFL/PRERED/VST/reduced/' + field_name + '_wide_field/'

    #middlelink = './data2/'
    #allothers = './data3/'

    obsdate = obs_date #corrected for LST difference
    date = datetime.strftime(datetime.strptime(obsdate, '%Y-%m-%d')-timedelta(days=1), '%Y-%m-%d')

    finalseepoch = np.sort(glob.glob(allothers+'*'+obsdate+'*.fits')) #[:1]
    eachtimes = np.unique([epochname.split(obsdate)[1].split('_')[0] for epochname in finalseepoch])
    coadd = fits.open(glob.glob(middlelink+'/mosaic/*'+date+'*.fits')[0])
    names = []

    for aaa in range(len(eachtimes)):

        singleepochs = glob.glob(allothers+'*'+obsdate+eachtimes[aaa]+'*.fits')
        finalchip = glob.glob(middlelink+'*'+obsdate+eachtimes[aaa]+'*.fits')
        allepochs = np.array(finalchip+list(singleepochs))

        #for epoch in allepochs:
        #    print('scp lemon@login01.astro.unige.ch:'+epoch+' ./data3/')
        #print('scp lemon@login01.astro.unige.ch:'+glob.glob(middlelink+'mosaic/*'+date+'*.fits')[0]+' ./data2/')

        #allepochs  = glob.glob(allothers+'/*')

        all_hdus = []
        for epoch in allepochs:
            all_hdus.append(fits.open(epoch)[0])

        #array, footprint = reproject_interp(hdu2, coadd.header)
        print(all_hdus)
        from astropy import units as u
        wcs_out, shape_out = find_optimal_celestial_wcs(all_hdus, resolution=2.14 * u.arcsec)
        print(wcs_out)
        array, footprint = reproject_and_coadd(all_hdus, wcs_out, shape_out=shape_out,
                                               reproject_function=reproject_interp)

        datestring = [allepochs[0].split('/')[-1].split('_')[0][6:]]
        starttime = Time(datestring, format='isot', scale='utc').mjd[0]
        exptime = all_hdus[0].header['EXPTIME']/(24.*3600.)
        endtime = starttime+exptime

        header = wcs_out.to_header()
        primary_hdu = fits.PrimaryHDU(array, header=header)
        primary_hdu.header['STARTMJD'] = starttime
        primary_hdu.header['ENDMJD'] = endtime

        #hdu = fits.ImageHDU(array)

        hdul = fits.HDUList([primary_hdu])

        name = allepochs[0].split('/')[-1].split('_')[0]+'_fullfield_binned.fits'
        hdul.writeto(path+name, overwrite=True)

        names.append(name)

    return names

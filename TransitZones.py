"""
This code reproduces the figures and tables in the R. Wells et al. 2017 paper.
The code is structured into different sections for each separate code block.
Parameters for the code should be set below. Set to 0 to not run and set to 1 to run the related code block.
"""

# Code parameters - 0/1
plot_ecl = 0            # plots all transit zones - reproduces figure 2 in the paper
plot_k2 = 0             # overplots K2 fields - reproduces figure 5 in the paper
plot_gal = 0            # plots orbits of planets over the galactic plane
find_crossovers = 0     # finds intersection points between all transit zones
                        # - set "eq" below to run using the transit zone angle, grazing angle or approximation
plot_intersects = 0     # plots each crossover individually while searching
plot_subplot = 0        # plots 2 crossover regions - reproduces figure 3 in the paper

print_region_corner_table = 0   # outputs the table of crossover regions corners - reproduces appendix 1
find_planets = 0                # finds all known planets which fall into transit zones -reproduces appendix 3
print_probabilities = 0         # outputs table of all transiting probabilities - reproduces appendix 2
print_comparison = 0            # outputs table comparing sizes of crossover regions - reproduces table 2
plot_comparison = 0             # plots comparison of a crossover region size - reproduces figure 4 in paper
                                # - set "comp_region" below to choose which crossover. e.g. 4,5: Jupiter, Saturn
                                # - set "plot_cs" below to choose which angles to compare. e.g. tz angle to approx
print_planet_params = 0         # outputs table of TZ & grazing angle, plus transit depths - reproduces table 1

dens = 400                  # number of data points / degree in fits
eq = 't'                    # run crossover code for transit zone angle, approximation or grazing angle ('t', 'a', 'g')
comp_region = [4, 5]        # regions to plot in comparison, 0:Mercury - 7:Neptune
plot_cs = [0, 2]            # which angles to compare - 0:transit zone, 1:approximation, 2:grazing

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
"""Code setup"""

if plot_k2 == 1 and plot_ecl == 0:              # plot_k2 requires plot_ecl
    plot_ecl = 1

import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.cm as cm
import os
from scipy.optimize import curve_fit, fsolve
from itertools import combinations
import pickle
import sys
import matplotlib.image as mpimg
import pandas

for directory in ['FigOut', 'DataOut']:         # create output directories if not present
    if not os.path.exists(directory):
        os.makedirs(directory)
pandas.set_option('chained_assignment', None)   # turn off pandas copy warning

def TZ_calc(R_p, a, R_s):
    """
    Calculates the transit zone angle
    :param R_p: Planetary radius
    :param a: Sun-planet distance
    :param R_s: Solar radius
    :return: Transit zone angle
    """
    return np.degrees(2.0 * (np.arctan(R_s / a) - np.arcsin(R_p / np.sqrt(a*a + R_s*R_s))))


def graze_calc(R_p, a, R_s):            # calculates grazing angle
    return np.degrees(2.0 * np.arctan((R_p + R_s) / a))


def approx_TZ_calc(a, R_s):
    return np.degrees(2.0 * R_s / a)


def fit(x, A, d):                       # fits sine curve
    x = np.radians(x)

    return A * np.sin(x - d)


def fit2(x, A, d, c):                   # fits sine curve + offset
    x = np.radians(x)

    return A * np.sin(x - d) + c


def nfit(x, A1, d1, A2, d2, c, sign):   # transit zone equation to solve for crossovers
    x = np.radians(x)

    return A1 * np.sin(x - d1) + sign * (A2 * np.sin(x - d2) + c) / 2.0


def sd_to_prob(sd):              # converts square degrees to a transiting probability
    return sd * (np.pi / 180.0)**2.0 / (4.0 * np.pi)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
"""Loads planetary data and compute angles"""

# data from http://solarsystem.nasa.gov/planets/
au = 149597870700.0 / 1000.0    # 1 AU (km)
R_sun = 695508.0                # Radius of Sun (km)
names = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']     # names of planets
radii = [2.4397E3, 6.0518E3, 6.3710E3, 3.3895E3, 6.9911E4, 5.8232E4, 2.5362E4, 2.4622E4]    # radii of planets
s_d = [57.909227E6, 1.0820948E8, 1.4959826E8, 2.2794382E8, 7.7834082E8, 1.4266664E9,        # semi-major axis
       2.8706582E9, 4.4983964E9]

colors = cm.rainbow(np.linspace(0, 1, len(names)))              # range of colours for plotting
cols = ['darkmagenta', 'darkolivegreen', 'darkgoldenrod']       # colours for plotting

sun_distances = []          # Sun-planet distances over 1 complete orbit from JPL Horizons
for i in range(len(names)):
    a = np.genfromtxt('OrbitData/ecl_helio_'+names[i]+'.txt', delimiter=',', skip_header=34, skip_footer=50)[:, 8]
    sun_distances.append(a)

psi_TZ = []             # transit zone angle
graze = []              # grazing angle
transit_depth = []      # transit depth
psi_TZ_ar = []          # variable transit zone angle over 1 orbit
graze_ar = []           # variable grazing angle over 1 orbit
approx_ar = []          # variable approximated angle over 1 orbit
for i in range(len(names)):
    R = radii[i]                # planetary radius
    d = sun_distances[i]        # semi-major axis

    # compute angles over 1 complete orbit
    approx_ar.append([])
    psi_TZ_ar.append([])
    graze_ar.append([])
    for j in range(len(d)):
        psi_TZ_ar[i].append(TZ_calc(R, d[j]*au, R_sun))
        approx_ar[i].append(approx_TZ_calc(d[j] * au, R_sun))
        graze_ar[i].append(graze_calc(R, d[j] * au, R_sun))

    psi = TZ_calc(R, s_d[i], R_sun)
    psi_TZ.append(psi)

    td = R**2 / R_sun**2
    transit_depth.append(td)

    grz = graze_calc(R, s_d[i], R_sun)
    graze.append(grz)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
"""Outputs table of planetary parameters"""

if print_planet_params == 1:
    tab1str = 'Planet,TZ angle,Grazing angle,% diff,Transit depth\n'        # construct string
    for i in range(len(names)):
        diff_grz = round((graze[i] - psi_TZ[i]) / psi_TZ[i] * 100.0, 1)     # % difference between tz angle and approx

        tab1str += names[i] + ',' + str(format(psi_TZ[i], '.4f')) + ',' + str(format(graze[i], '.4f')) + ',' + \
                   str(diff_grz) + ',' + str(format(transit_depth[i]*100.0, '.4f')) + '\n'

    with open('DataOut/PlanetParameters.csv', 'w') as f:                    # save to file
        f.write(tab1str)
    print '> Planet parameters table saved to \"DataOut/PlanetParameters.csv\".'

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
"""Loads JPL Horizons data and makes any needed files"""

# Load ecliptic data from JPL Horizons
ecl_lon_list, ecl_lat_list = [], []     # helio-centric ecliptic coordinates of the solar system planets over 1 orbit
for i in range(len(names)):
    ecl_lon_list.append(np.genfromtxt('OrbitData/ecl_helio_' + names[i] + '.txt', delimiter=',', skip_header=34,
                                      skip_footer=50)[:, 6])
    ecl_lat_list.append(np.genfromtxt('OrbitData/ecl_helio_' + names[i] + '.txt', delimiter=',', skip_header=34,
                                      skip_footer=50)[:, 7])

# Make galactic coordinates file
if plot_gal == 1:
    exists = []
    for i in range(len(names)):
        exists.append(os.path.isfile('OrbitData/gal_' + names[i] + '.txt'))
    gal_files_exist = set(exists) == {True}     # do files exist already?

    if not gal_files_exist:                     # if not, then make them
        print '> Making galactic coordinate files.'
        for i in range(len(names)):
            print '> >', i + 1, '/', len(names)
            txt_file = 'OrbitData/gal_' + names[i] + '.txt'
            with open(txt_file, 'w') as f:
                for j in range(len(ecl_lon_list[i])):       # convert to galactic coordinate system
                    g = SkyCoord(ecl_lon_list[i][j], ecl_lat_list[i][j], unit=(u.degree, u.degree),
                                 distance=30.0*u.lyr, frame='heliocentrictrueecliptic', equinox='J2000.0').galactic
                    f.write(str(g.l.degree) + '\t' + str(g.b.degree) + '\n')    # save to file

# Make ecliptic K2 fields file
if plot_k2 == 1:
    # K2 field coordinates (equatorial) from https://keplerscience.arc.nasa.gov/k2-fields.html#machine-readable-files
    d = np.genfromtxt('k2-footprint.csv', delimiter=',', skip_header=1)

    cs_inds = []    # first coordinate of campaigns for annotating
    for i in range(17):
        cs_inds.append(list(d[:, 0]).index(i))

    if not os.path.exists('DataIn/K2_fields_ecliptic.pkl'):         # convert to ecliptic if file not present
        print '> Making ecliptic K2 fields file.'
        lon_l, lat_l = [], []

        for j in range(len(d)):
            rr, dd = [], []
            for i in range(4):  # convert to ecliptic
                ecl_c = SkyCoord(d[j][6 + 2 * i], d[j][7 + 2 * i], unit=(u.degree, u.degree), frame='icrs',
                                 equinox='J2000.0', distance=30.0 * u.lyr).heliocentrictrueecliptic

                rr.append(ecl_c.lon.degree)
                dd.append(ecl_c.lat.degree)
            lon_l.append(rr)
            lat_l.append(dd)

            if j % 100 == 0 or j+1 == len(d):
                print '> >', j, '/', len(d)

        with open('DataIn/K2_fields_ecliptic.pkl', 'wb') as f:      # save to pickle file
            pickle.dump([lon_l, lat_l], f)

    else:       # load pickle file
        with open('DataIn/K2_fields_ecliptic.pkl', 'rb') as f:
            lon_l, lat_l = pickle.load(f)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
"""Fits sinusoidal curves to the ecliptic coordinates and variable angles over 1 complete orbit
   The parameter 'dens' gives the density of the curves. I.e. 'dens' datapoints per degree"""

print '> Fitting curves to data. (dens = ' + str(dens) + ')'
data_fits = []      # holds all fits to the coordinates
fit_params = []     # holds all parameters of each fit
psi_fits = []       # transit zone angle
psi_params = []
graze_fits = []     # grazing angle
graze_params = []
approx_fits = []    # approximate angle
approx_params = []

if plot_ecl == 1:
    fig = plt.figure(figsize=(15, 7))       # initialise figure
    ax = fig.add_subplot(111)
    plt.minorticks_on()

for i in range(len(names)):
    popt1, pcov1 = curve_fit(fit, ecl_lon_list[i], ecl_lat_list[i])     # fit coordinates to sine curve
    fit_params.append(popt1)

    popt2, pcov2 = curve_fit(fit2, ecl_lon_list[i], psi_TZ_ar[i])       # transit zone angle
    psi_params.append(popt2)

    popt3, pcov3 = curve_fit(fit2, ecl_lon_list[i], graze_ar[i])        # grazing angle
    graze_params.append(popt3)

    popt4, pcov4 = curve_fit(fit2, ecl_lon_list[i], approx_ar[i])       # approximate angle
    approx_params.append(popt4)

    data_fit = []
    psi_fit = []
    graze_fit = []
    approx_fit = []
    x_fit = []      # longitude for fit

    for j in range(360 * dens):
        data_fit.append(fit(j / float(dens), popt1[0], popt1[1]))
        psi_fit.append(fit2(j / float(dens), popt2[0], popt2[1], popt2[2]))
        graze_fit.append(fit2(j / float(dens), popt3[0], popt3[1], popt3[2]))
        approx_fit.append(fit2(j / float(dens), popt4[0], popt4[1], popt4[2]))
        x_fit.append(j / float(dens))

    approx_fits.append(approx_fit)
    psi_fits.append(psi_fit)
    data_fits.append(data_fit)
    graze_fits.append(graze_fit)

    if plot_ecl == 1:
        if i != 2:          # colours on plot - Earth as black
            c = colors[i]
        else:
            c = 'black'
        df1 = data_fit + np.asarray(psi_fits[i]) / 2.0      # upper transit zone boundary
        df2 = data_fit - np.asarray(psi_fits[i]) / 2.0      # lower transit zone boundary

        ax.fill_between(x_fit, df1, df2, where=df1 >= df2, edgecolor=c, facecolor=c,    # plot zones as coloured bands
                        alpha=0.4, interpolate=True, label=names[i])

if plot_ecl == 1:
    if plot_k2 == 1:
        for i in range(len(d)):             # plot areas between corner points of detectors
            plt.fill(lon_l[i], lat_l[i], edgecolor='grey', facecolor='grey', alpha=0.5, zorder=1)

        for i in range(len(cs_inds)):       # annotate campaign numbers
            campaign_start = [lon_l[cs_inds[i]][0], lat_l[cs_inds[i]][0]]
            plt.annotate(i, xy=campaign_start, xytext=(campaign_start[0] - 15, campaign_start[1]), fontsize=20,
                         color='r')

        figname = 'FigOut/AllHelioEcliptic+K2Fields.png'
    else:
        figname = 'FigOut/AllHelioEcliptic.png'

    ax.set_xlabel('Longitude (Degrees)', fontsize=15)
    ax.set_ylabel('Latitude (Degrees)', fontsize=15)
    ax.set_xlim(0.0, 360.0)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.tick_params(axis='both', which='both', width=2)
    plt.tick_params(axis='both', which='major', length=7)
    plt.tick_params(axis='both', which='minor', length=4)
    ax.legend(loc=1)
    plt.savefig(figname, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    print '> Transit zones plot saved to \"'+figname+'\".'

if plot_gal == 1:
    fig = plt.figure(figsize=(10.0, 5.0))
    ax = fig.add_subplot(111, projection='mollweide')       # make mollweide projection plot
    ax.grid(color='darkgrey', lw=2)
    img = mpimg.imread('DataIn/GalaxyImage.png')            # plot over Mellinger + 2MASS galaxy image
    ax.imshow(img, extent=[-np.pi, np.pi, -np.pi / 2.0, np.pi / 2.0], aspect='auto')

    for i in range(len(names)):
        gal_file = np.genfromtxt('OrbitData/gal_' + names[i] + '.txt', delimiter='\t')
        gal_l = gal_file[:, 0] - 180.0      # galactic longitude
        gal_b = gal_file[:, 1]              # galactic latitude
        gal_l, gal_b = zip(*sorted(zip(np.radians(gal_l), np.radians(gal_b))))

        plt.plot(gal_l, gal_b, color='lightgrey', lw=2, zorder=5)

    plt.xlabel('Longitude (degrees)', fontsize=15)
    plt.ylabel('Latitude (degrees)', fontsize=15)
    [i.set_color("darkgrey") for i in plt.gca().get_xticklabels()]
    ax.tick_params(labelsize=15)
    plt.savefig('FigOut/Galactic_orbits.png', format='png', dpi=200, bbox_inches='tight', pad_inches=0)
    print '> Galactic plot saved to \"FigOut/Galactic_orbits.png\".'

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
"""Finds intersection points of the transit zones"""

if find_crossovers == 1:
    if eq == 't':                       # load data for specified angle
        x_ar = np.asarray(psi_fits)
        para = psi_params
    elif eq == 'a':
        x_ar = np.asarray(approx_fits)
        para = approx_params
    elif eq == 'g':
        x_ar = np.asarray(graze_fits)
        para = graze_params
    else:
        print '>> Bad eq (t, a, g)'
        sys.exit()

    print '> Looking for crossovers between transit zones.'
    planet_inds = np.linspace(0, 7, 8, dtype='int')    # planet ind numbers
    pls_list = []       # holds all combinations of planet_inds (2 & 3)
    for i in combinations(planet_inds, 2):      # all combinations of 2 planets
        pls_list.append(i)
    for i in combinations(planet_inds, 3):      # all combinations of 3 planets
        pls_list.append(i)

    region_pls_list, region_lon_list, region_lat_list = [], [], []    # hold all region corner point coordinates

    # load indices of intersect points - found by running with plot_intersects=1
    c3file = np.genfromtxt('DataIn/3-crossover_regions.txt', delimiter='\t', dtype='string')
    c3pls = []
    c3points = []
    for p in c3file:
        intpts = []
        c3pls.append(p[0])
        if p[1] == '-':
            c3points.append('-')
        else:
            for i in p[1].split(','):
                intpts.append(int(i))
            c3points.append(intpts)

    sp = []     # for subplot code
    t = np.linspace(0.0, 350.0, 36)     # spacing to search for crossovers

    for pls in pls_list:    # loop over all combinations
        n = len(pls)        # number of planets (1, 2 or 3)

        using_names = ''    # names of planets being used
        for i in range(n):
            using_names += names[pls[i]]
            if i != n - 1:
                using_names += ', '
        print '> > Looking for intersects between:', using_names

        if n != 2 and n != 3:     # no 4+ regions exist
            print 'n is not a usable number. n =', n
            sys.exit()

        # sine function fits to data +/- psi
        l = []
        for i in pls:
            l.append(data_fits[i] + x_ar[i] / 2.0)      # upper transit zone boundary
            l.append(data_fits[i] - x_ar[i] / 2.0)      # lower transit zone boundary

        # find intersection points - 2 for each
        x_list = []     # longitude of intersect
        y_list = []     # latitude of intersect
        idcs = []
        done = []
        for i in range(n*2):            # loop over all boundaries with all boundaries
            for j in range(n*2):
                ieven = i % 2 == 0      # True / False
                jeven = j % 2 == 0
                if ieven:
                    i_range = [i, i + 1]        # indices of first zone's boundaries
                    isign = 1                   # plus angle/2
                    pli = pls[i / 2]            # indice of planet 1
                else:
                    i_range = [i - 1, i]
                    isign = -1                  # minus angle/2
                    pli = pls[(i - 1) / 2]
                if jeven:
                    jsign = 1                   # plus angle/2
                    plj = pls[j / 2]            # indice of planet 2
                else:
                    jsign = -1                  # minus angle/2
                    plj = pls[(j - 1) / 2]

                rev_str = str(j) + str(i)       # reverse of indices to check if done already

                if j not in i_range and rev_str not in done:   # if not the same planet and combination not already done
                    # solve for crossover points at each t step
                    fs = fsolve(lambda x: nfit(x, fit_params[pli][0], fit_params[pli][1], para[pli][0],
                                               para[pli][1], para[pli][2], isign) -
                                          nfit(x, fit_params[plj][0], fit_params[plj][1], para[plj][0],
                                               para[plj][1], para[plj][2], jsign), t)

                    # get unique between 0 and 360 degrees only
                    res = []
                    for k in fs:
                        if 0.0 <= k < 360.0:
                            res.append(round(k, 4))
                    dmy = list(set(res))
                    fs_x = []
                    for dm in range(-1, len(dmy)-1):
                        if not dmy[dm-1]-0.0002 < dmy[dm] < dmy[dm+1]+0.0002:
                            fs_x.append(dmy[dm])
                    x_list += fs_x

                    for x in fs_x:     # latitude of crossover points
                        y_list.append(round(nfit(x, fit_params[pli][0], fit_params[pli][1], para[pli][0],
                                                 para[pli][1], para[pli][2], isign), 4))
                    idcs.append([i, j])
                    done.append(str(i) + str(j))

        x_list, y_list = zip(*sorted(zip(x_list, y_list)))      # order lists by longitude

        if plot_intersects == 1:     # for finding region corners by eye
            plt.plot(x_list, y_list, marker='o', ls='', color='black')     # plot points
            for i in range(len(x_list)):                                   # annotate index in list
                plt.annotate(str(i), (x_list[i], y_list[i]), fontsize=15, color='black')

        if n == 3:      # 3-planet crossover
            # get fill region
            y1 = np.array(map(min, zip(l[2], l[4])))
            y2 = np.array(map(max, zip(l[1], l[5])))
            y3 = np.array(map(min, zip(y1, l[0])))
            y4 = np.array(map(max, zip(y2, l[3])))
            plt.fill_between(x_fit, y3, y4, where=y3 >= y4, color='grey', alpha=0.8)
            # plt.show()

            plsstr = ''                  # string of pls, e.g. '123'
            for i in pls:
                plsstr += str(i)

            idx = c3pls.index(plsstr)    # get index of permutation in file
            point_inds = c3points[idx]   # indices of points in x_ & y_lists

            if point_inds != '-':        # if crossover exists
                xx, yy = [], []
                for i in point_inds:
                    xx.append(x_list[i])
                    yy.append(y_list[i])
                is_region = True
            else:                        # if no crossover
                is_region = False

        elif n == 2:        # 2-planet crossover
            # get fill region
            y1 = np.array(map(min, zip(l[0], l[2])))
            y2 = np.array(map(max, zip(l[1], l[3])))
            plt.fill_between(x_fit, y1, y2, where=y1 >= y2, color='grey', alpha=0.8)
            # plt.show()

            regionsx, regionsy = [[], []], [[], []]     # separate regions
            v = 60.0                                    # maximum spacing between points

            if list(pls) == [4,7]:      # Jupiter-Neptune region crosses 0/360 line
                for i in range(len(x_list)):
                    if x_list[2] - v <= x_list[i] <= x_list[2] + v:
                        regionsx[0].append(x_list[i])
                        regionsy[0].append(y_list[i])
                    else:
                        regionsx[1].append(x_list[i])
                        regionsy[1].append(y_list[i])

                x0, y0 = regionsx[1][0], regionsy[1][0]
                del regionsx[1][0]
                del regionsy[1][0]
                regionsx[1].append(x0)
                regionsy[1].append(y0)

            else:
                for i in range(len(x_list)):
                    if x_list[5] - v <= x_list[i] <= x_list[5] + v:
                        regionsx[1].append(x_list[i])
                        regionsy[1].append(y_list[i])
                    else:
                        regionsx[0].append(x_list[i])
                        regionsy[0].append(y_list[i])

            xx = regionsx[0] + regionsx[1]
            yy = regionsy[0] + regionsy[1]
            is_region = True

        if is_region is False:
            print '> > > Region not found.'

        else:
            print '> > >', len(xx), 'corners found.'
            # save region corner coordinates
            region_pls_list.append(pls)
            region_lon_list.append(xx)
            region_lat_list.append(yy)

        if plot_intersects == 1:
            for i in range(len(l)):
                cid = (i - (i % 2)) / 2
                col = cols[cid]
                ls = '-'
                if i % 2 == 0:
                    lab = names[pls[i/2]]
                else:
                    lab = None
                plt.plot(x_fit, l[i], color=col, label=lab, ls=ls, lw=1.5)      # plot boundaries

            if is_region:
                plt.plot(xx, yy, 'o', color='black')        # plot intersection points

            plt.xlim(0, 360)
            plt.legend()
            plt.xlabel('Longitude (Degrees)')
            plt.ylabel('Latitude (Degrees)')
            plt.title('Regions of the galaxy which can detect transits of '+using_names)
            plt.tick_params(axis='both', which='both', width=2)
            plt.tick_params(axis='both', which='major', length=7)
            plt.tick_params(axis='both', which='minor', length=4)

            if is_region:
                plt.show()
            else:
                plt.clf()

        if pls == (3, 4) or pls == (0, 2, 3) and eq == 't':     # save for regions subplot (plot_subplot=1)
            sp.append([])
            sp[-1].append(pls)
            sp[-1].append(xx)
            sp[-1].append(yy)
            sp[-1].append(l)
            if n == 2:
                sp[-1].append([y1, y2])
            else:
                sp[-1].append([y3, y4])

    all = [region_pls_list, region_lon_list, region_lat_list]   # save to pickle file for table
    with open('DataIn/all_region_corner_points_'+eq+'.pkl', 'wb') as f:
        pickle.dump(all, f)

    if eq == 't':
        sp.append(x_fit)
        with open('DataIn/regions_subplot.pkl', 'wb') as f:         # for subplot of regions
            pickle.dump(sp, f)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
"""Plots 2 regions in one figure (figure 3 in paper)
   Requires 'DataIn/regions_subplot.pkl' file from running find_crossovers=1 with eq='t'"""

if plot_subplot == 1:
    with open('DataIn/regions_subplot.pkl', 'rb') as f:     # load subplot data
        sp = pickle.load(f)

    x_fit = sp[2]

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))     # initialise figure

    for i in range(2):
        pls = sp[i][0]          # planets in crossover
        xx = sp[i][1]           # logitude of intersect points
        yy = sp[i][2]           # latitude of intersect points
        l = sp[i][3]            # all transit zone boundaries
        y1, y2 = sp[i][4]       # borders of crossover region

        for j in range(len(l)):
            cid = (j - (j % 2)) / 2
            col = cols[cid]
            if j % 2 == 0:
                lab = names[pls[j/2]]
            else:
                lab = None

            if i == 0:
                ax1.plot(x_fit, l[j], color=col, label=lab, lw=3, alpha=1)          # plot transit zone boundaries
                ax1.set_xlim(170, 200)
                ax1.set_ylim(1.2, 1.4)
                ax1.plot(xx, yy, marker='o', markersize=6, ls='', color='black')    # plot intersection points
            else:
                ax2.plot(x_fit, l[j], color=col, label=lab, lw=3, alpha=1)
                ax2.set_xlim(35, 60)
                ax2.set_ylim(-0.5, 0.5)
                ax2.plot(xx, yy, marker='o', markersize=6, ls='', color='black')

        if i == 0:
            ax1.fill_between(x_fit, y1, y2, where=y1 >= y2, color='grey', alpha=0.6)    # shade crossover region
        else:
            ax2.fill_between(x_fit, y1, y2, where=y1 >= y2, color='grey', alpha=0.6)

    ax1.minorticks_on()
    ax2.minorticks_on()
    ax1.legend()
    ax2.legend()
    f.text(0.5, 0.04, 'Longitude (Degrees)', ha='center', va='center', fontsize=15)
    ax1.set_ylabel('Latitude (Degrees)', fontsize=15)
    ax1.tick_params(axis='both', which='both', width=2)
    ax1.tick_params(axis='both', which='major', length=7)
    ax1.tick_params(axis='both', which='minor', length=4)
    ax2.tick_params(axis='both', which='both', width=2)
    ax2.tick_params(axis='both', which='major', length=7)
    ax2.tick_params(axis='both', which='minor', length=4)
    plt.savefig('FigOut/RegionsSubplot.eps', format='eps', dpi=1000, bbox_inches='tight', pad_inches=0)
    print '> Subplot saved to \"FigOut/RegionsSubplot.eps\".'

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- -
"""Outputs a table of all crossover intersection points using the transit zone angle
   Reproduces appendix 1 in the paper"""

if print_region_corner_table == 1:
    with open('DataIn/all_region_corner_points_t.pkl', 'rb') as f:      # load regions
        planets, lon, lat = pickle.load(f)

    ra, dec = [], []                # equatorial coordinates of intersection points
    for i in range(len(lon)):
        ra.append([])
        dec.append([])
        for j in range(len(lon[i])):
            eq = SkyCoord(float(lon[i][j]), float(lat[i][j]), unit=(u.degree, u.degree),    # convert to equatorial
                          frame='heliocentrictrueecliptic', equinox='J2000.0', distance=30.0 * u.lyr).icrs
            ra[i].append(format(eq.ra.degree, '.4f'))       # 4 dp
            dec[i].append(format(eq.dec.degree, '.4f'))

    for i in range(len(ra)):        # order regions
        n = len(ra[i]) / 2
        reg1 = [ra[i][:n], dec[i][:n]]
        reg2 = [ra[i][n:], dec[i][n:]]
        reg1[0], reg1[1] = zip(*sorted(zip(reg1[0], reg1[1])))
        reg2[0], reg2[1] = zip(*sorted(zip(reg2[0], reg2[1])))

        ra[i] = reg1[0] + reg2[0]
        dec[i] = reg1[1] + reg2[1]

    string = ''
    for i in range(len(planets)):       # loop all regions
        n_pls = len(planets[i])         # number of planets in region

        for j in range(n_pls):          # planets in region
            pl_ind = planets[i][j]
            string += names[pl_ind]
            if j != n_pls - 1:
                string += '; '
            elif j == n_pls - 1:
                string += ','

        if i == 31:                     # Me, E, U (3 points, 1 region)
            n_pts = 3

            for j in range(n_pts):      # longitude
                string += str(ra[i][j])
                if j != n_pts - 1:
                    string += '; '
                else:
                    string += ','

            for j in range(n_pts):      # latitude
                string += str(dec[i][j])
                if j != n_pts - 1:
                    string += '; '
                else:
                    string += ','

        else:
            n_pts = len(ra[i]) / 2

            # region 1
            for j in range(n_pts):      # longitude
                string += str(ra[i][j])
                if j != n_pts - 1:
                    string += '; '
                else:
                    string += ','

            for j in range(n_pts):      # latitude
                string += str(dec[i][j])
                if j != n_pts - 1:
                    string += '; '
                else:
                    string += ','

            string += '\n,'             # next line for second region

            # region 2
            for j in range(n_pts, n_pts * 2):   # longitude
                string += str(ra[i][j])
                if j != n_pts * 2 - 1:
                    string += '; '
                else:
                    string += ','

            for j in range(n_pts, n_pts * 2):   # latitude
                string += str(dec[i][j])
                if j != n_pts * 2 - 1:
                    string += '; '

        string += '\n'

    with open('DataOut/RegionCornerTable.csv', 'w') as f:       # save table to file
        f.write(string)
    print '> Crossover corner table saved to \"DataOut/RegionCornerTable.csv\"'

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- --
"""Checks coordinates of known exoplanets to identify ones which fall into one or more transit zones
   Reproduces appendix 3 from the paper"""

if find_planets == 1:
    print '> Searching for exoplanets in transit zones.'
    # Load file containing all known exoplanets from http://exoplanet.eu/catalog/
    df = pandas.read_csv('DataIn/exoplanet.eu_catalog.csv', delimiter=',', low_memory=False)

    start_ind, end_ind = 0, len(df['# name'])           # go through entire file
    name, status, ecl_lon, ecl_lat, regions, total, radec, detect, mass, radius, period = \
        [], [], [], [], [], [], [], [], [], [], []      # information to save for each planet in a transit zone

    for j in range(start_ind, end_ind):         # go through csv file
        try:
            ra_dec = [df['ra'][j], df['dec'][j]]        # RA and Dec of planet

            ecl_c = SkyCoord(ra_dec[0], ra_dec[1], unit=(u.degree, u.degree), frame='icrs', equinox='J2000.0',
                             distance=30.0*u.lyr).heliocentrictrueecliptic      # convert to ecliptic
            ecliptic_lon = ecl_c.lon.degree                                     # ecliptic coordinates
            ecliptic_lat = ecl_c.lat.degree

            in_regs = []        # in regions, e.g. [Earth, Mars]
            for i in range(len(names)):         # check if in each planet's transit zone
                A1 = fit_params[i][0]           # load fits to transit zones and angles
                d1 = fit_params[i][1]
                A2 = psi_params[i][0]
                d2 = psi_params[i][1]
                c = psi_params[i][2]

                upper = fit(ecliptic_lon, A1, d1) + fit2(ecliptic_lon, A2, d2, c) / 2.0     # max longitude
                lower = fit(ecliptic_lon, A1, d1) - fit2(ecliptic_lon, A2, d2, c) / 2.0     # min longitude

                if lower <= ecliptic_lat <= upper:      # if within transit zone
                    in_regs.append(names[i])

            if len(in_regs) > 0:                        # if in 1 or more transit zones, save info
                name.append(df['# name'][j])
                status.append(df['planet_status'][j])
                ecl_lon.append(ecliptic_lon)
                ecl_lat.append(ecliptic_lat)
                regions.append(in_regs)
                total.append(len(in_regs))
                radec.append(ra_dec)
                detect.append(df['detection_type'][j])
                mass.append(df['mass'][j])
                radius.append(df['radius'][j])
                period.append(df['orbital_period'][j])

        except ValueError:
            print j, '= error\n'

        if j % 100 == 0 or j+1 == end_ind:
            print '> >', j, '/', end_ind

    n_conf, n_unconf, n_cand = status.count('Confirmed'), status.count('Unconfirmed'), status.count('Candidate')
    print '> > Found:', n_conf, 'confirmed,', n_unconf, 'unconfirmed and', n_cand, 'candidates.'

    string = 'Name,Status,RA/Dec,Zones,Total,Mass,Radius,Period\n'
    for i in range(len(name)):      # construct table
        string += name[i] + ',' + status[i] + ',' + str(format(radec[i][0], '.4f')) + str(format(radec[i][1], '.4f'))\
                  + ',' + ';'.join(regions[i]) + ',' + str(total[i]) + ',' + str(mass[i]) + ',' + str(radius[i]) +\
                  ',' + str(period[i]) + '\n'
    with open('DataOut/PlanetsInZones.csv', 'w') as f:      # save table to file
        f.write(string)
    print '> Known exoplanets in transit zones table saved to \"DataOut/PlanetsInZones.csv\".'

    # for i in range(8):
    #     if i == 2:
    #         c = 'black'
    #     else:
    #         c = colors[i]
    #     df1 = data_fits[i] + np.asarray(psi_fits[i]) / 2.0
    #     df2 = data_fits[i] - np.asarray(psi_fits[i]) / 2.0
    #     plt.fill_between(x_fit, df1, df2, where=df1 >= df2, edgecolor=c, facecolor=c, alpha=0.4, interpolate=True,
    #                      label=names[i])
    # plt.plot(ecl_lon, ecl_lat, 'o', color='black')
    # plt.xlim(0, 360)
    # plt.legend()
    # plt.show()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- --
"""Computes geometric transit probabilities for all transit zones and crossover regions
   print_probabilities and print_comparison reproduce appendix 2 and table 2 from the paper respectively
   requires .pkl files made by running find_crossovers=1 with eq='t', 'a' and 'g'"""

if print_probabilities == 1 or print_comparison == 1 or plot_comparison == 1:
    print '> Computing geometric transit probabilities.'
    x_fit = np.asarray(x_fit)                   # longitude of fits
    prob_str = 'Set,P,P/P_Earth\n'              # for probability table
    comp_str = 'Set,P_TZ,P_approx,% diff\n'     # for comparison table

    if print_probabilities == 0 and print_comparison == 0:
        pls_list = [tuple(comp_region)]

    else:
        planet_inds = np.linspace(0, 7, 8, dtype='int')     # [0 - 7]
        pls_list = []           # holds all combinations of planet_inds
        for i in range(8):      # list of single planets
            pls_list.append([i])
        for i in combinations(planet_inds, 2):      # all combinations of 2 planets
            pls_list.append(i)
        for i in combinations(planet_inds, 3):      # all combinations of 3 planets
            pls_list.append(i)

    # parameters for comparison plot
    tag_cols = ['grey', 'mediumseagreen', 'mediumblue']
    tag_labels = ['TZ', 'Approx', 'Graze']
    tag_alphas = [0.8, 0.3, 0.3]

    # sets of planets to compare for table
    comp_sets = [[0, 1], [1, 2], [2, 3], [0, 2, 3], [0, 1, 5], [0, 2, 6], [1, 2, 6], [0, 3, 6], [4, 5, 6], [0, 1, 7],
                 [3, 4, 7]]
    at_earth = 0.00460783226838         # transit probability of Earth

    lon, lat = [], []
    for eq in ['t', 'a', 'g']:      # load intersection points for each angle
        with open('DataIn/all_region_corner_points_' + eq + '.pkl', 'rb') as f:  # load regions for each case
            all = pickle.load(f)
        planets = all[0]
        lon.append(all[1])
        lat.append(all[2])

    plt.figure(figsize=(9, 7))      # for comparison plot

    for pls in pls_list:        # loop over all combinations
        n = len(pls)            # number of planets in combination

        using_names = ''        # names of planets in combination
        for i in range(n):
            using_names += names[pls[i]]
            if i != n - 1:
                using_names += ';'

        # transit zone boundaries for each angle
        l = []      # tz
        l_a = []    # approx
        l_g = []    # graze
        for i in pls:
            l.append(data_fits[i] + np.asarray(psi_fits[i]) / 2.0)      # upper
            l.append(data_fits[i] - np.asarray(psi_fits[i]) / 2.0)      # lower
            l_a.append(data_fits[i] + np.asarray(approx_fits[i]) / 2.0)
            l_a.append(data_fits[i] - np.asarray(approx_fits[i]) / 2.0)
            l_g.append(data_fits[i] + np.asarray(graze_fits[i]) / 2.0)
            l_g.append(data_fits[i] - np.asarray(graze_fits[i]) / 2.0)

        if n == 1:      # single planet probabilities
            i = pls[0]
            at = sd_to_prob(np.trapz(l[0], x=x_fit) - np.trapz(l[1], x=x_fit))              # tz probability
            approx_at = sd_to_prob(np.trapz(l_a[0], x=x_fit) - np.trapz(l_a[1], x=x_fit))   # approx probability
            graze_at = sd_to_prob(np.trapz(l_g[0], x=x_fit) - np.trapz(l_g[1], x=x_fit))    # graze probability

            aptz_diff = round((approx_at - at) / at * 100.0, 1)     # approx/tz difference

            if print_comparison == 1:
                comp_str += using_names + ',' + '%.2e' % at + ',' + '%.2e' % approx_at + ',' + str(aptz_diff) + '\n'

            if print_probabilities == 1:
                prob_str += using_names + ',' + '%.2e' % at + ',' + '%.2e' % (at / at_earth) + '\n'

        else:       # if 2 or 3 planets
            try:    # print_comparison ValueErrors if not in list
                all_ind = planets.index(tuple(pls))     # position in pickle file lists
                npts = len(lon[0][all_ind]) / 2         # half length of region - i.e. 4 or 6

                if n == 3:      # 3 planet region boundaries
                    # get borders of crossover region
                    y1 = np.array(map(min, zip(l[2], l[4])))
                    y2 = np.array(map(max, zip(l[1], l[5])))
                    y3 = np.array(map(min, zip(y1, l[0])))
                    y4 = np.array(map(max, zip(y2, l[3])))
                    y1_a = np.array(map(min, zip(l_a[2], l_a[4])))
                    y2_a = np.array(map(max, zip(l_a[1], l_a[5])))
                    y3_a = np.array(map(min, zip(y1_a, l_a[0])))
                    y4_a = np.array(map(max, zip(y2_a, l_a[3])))
                    y1_g = np.array(map(min, zip(l_g[2], l_g[4])))
                    y2_g = np.array(map(max, zip(l_g[1], l_g[5])))
                    y3_g = np.array(map(min, zip(y1_g, l_g[0])))
                    y4_g = np.array(map(max, zip(y2_g, l_g[3])))
                    y_u, y_l = [y3, y3_a, y3_g], [y4, y4_a, y4_g]

                elif n == 2:    # 2 planet region boundaries
                    # get borders of crossover region
                    y1 = np.array(map(min, zip(l[0], l[2])))
                    y2 = np.array(map(max, zip(l[1], l[3])))
                    y1_a = np.array(map(min, zip(l_a[0], l_a[2])))
                    y2_a = np.array(map(max, zip(l_a[1], l_a[3])))
                    y1_g = np.array(map(min, zip(l_g[0], l_g[2])))
                    y2_g = np.array(map(max, zip(l_g[1], l_g[3])))
                    y_u, y_l = [y1, y1_a, y1_g], [y2, y2_a, y2_g]

                r_inds = []     # get start and end indices of regions
                for i in range(3):
                    r1_i1 = (np.abs(x_fit - lon[i][all_ind][0])).argmin()           # first region - start
                    r1_i2 = (np.abs(x_fit - lon[i][all_ind][npts - 1])).argmin()    # first region - end
                    r2_i1 = (np.abs(x_fit - lon[i][all_ind][npts])).argmin()        # second region - start
                    r2_i2 = (np.abs(x_fit - lon[i][all_ind][-1])).argmin()          # second region - end
                    r_inds.append([r1_i1, r1_i2, r2_i1, r2_i2])                     # list of indices

                at = np.zeros(3)        # holds probabilities for each angle

                for c in range(3):      # tz, approx, graze
                    if pls == tuple([0, 2, 6]):     # Me, E, U - only 1 region
                        i1 = (np.abs(x_fit - min(lon[c][all_ind]))).argmin()        # start of region
                        i2 = (np.abs(x_fit - max(lon[c][all_ind]))).argmin()        # end of region

                        upper = y_u[c][i1:i2]       # upper boundary
                        lower = y_l[c][i1:i2]       # lower boundary
                        x_cut = x_fit[i1:i2]        # x range of boundaries

                        # transit probability = area below upper border - area below lower border
                        at[c] += sd_to_prob(np.trapz(upper, x=x_cut) - np.trapz(lower, x=x_cut))

                        if c in plot_cs:        # comparison plot
                            plt.plot(x_cut, upper, color=tag_cols[c])
                            plt.plot(x_cut, lower, color=tag_cols[c])

                    else:
                        for i in [0, 2]:        # each region
                            if i == 0:
                                reg = lon[c][all_ind][:npts]    # region 1
                            else:
                                reg = lon[c][all_ind][npts:]    # region 2

                            if max(reg) - min(reg) < 100.0:     # doesn't cross 360
                                i1, i2 = r_inds[c][i], r_inds[c][i + 1]

                                upper = y_u[c][i1:i2]       # upper boundary
                                lower = y_l[c][i1:i2]       # lower boundary
                                x_cut = x_fit[i1:i2]        # x range of boundaries

                                at[c] += sd_to_prob(np.trapz(upper, x=x_cut) - np.trapz(lower, x=x_cut))  # probability

                            else:       # crosses 360
                                split1, split2 = [], []     # split region into near 0 and near 360
                                for x in reg:
                                    if x < 100.0:
                                        split1.append(x)
                                    else:
                                        split2.append(x)

                                # get end of near 0 region and start of near 360 region
                                s1max, s2min = (np.abs(x_fit - max(split1))).argmin(), \
                                               (np.abs(x_fit - min(split2))).argmin()

                                upper1, upper2 = y_u[c][:s1max], y_u[c][s2min:]     # upper boundary
                                lower1, lower2 = y_l[c][:s1max], y_l[c][s2min:]     # lower boundary
                                x_cut1, x_cut2 = x_fit[:s1max], x_fit[s2min:]       # x range of boundaries

                                at_s1 = np.trapz(upper1, x=x_cut1) - np.trapz(lower1, x=x_cut1)     # split 1 area
                                at_s2 = np.trapz(upper2, x=x_cut2) - np.trapz(lower2, x=x_cut2)     # split 2 area
                                at[c] += sd_to_prob(at_s1 + at_s2)      # probability

                if print_comparison == 1:
                    diff_tz_a = (at[0] - at[1]) / at[0] * 100.0     # % difference tz to approx

                    comp_str += using_names + ',' + '%.2e' % at[0] + ',' + '%.2e' % at[1] + ',' + \
                                str(round(diff_tz_a, 1)) + '\n'

                if print_probabilities == 1:
                    prob_str += using_names + ',' + '%.2e' % at[0] + ',' + '%.2e' % (at[0] / at_earth) + '\n'

                # comparison plot
                if list(comp_region) == list(pls):
                    for h in reversed(plot_cs):
                        plt.fill_between(x_fit, y_u[h], y_l[h], where=y_u[h] >= y_l[h], color=tag_cols[h],
                                         alpha=tag_alphas[h], label=tag_labels[h])
                    plt.xlim(min(x_cut) - 0.2, max(x_cut) + 0.2)
                    plt.ylim(min(lower) - 0.01, max(upper) + 0.01)
                    plt.xlabel('Longitude (Degrees)', fontsize=20)
                    plt.ylabel('Latitude (Degrees)', fontsize=20)
                    plt.legend(loc='best', fontsize=20)
                    plt.minorticks_on()
                    plt.tick_params(axis='both', which='both', width=2)
                    plt.tick_params(axis='both', which='major', length=7)
                    plt.tick_params(axis='both', which='minor', length=4)
                    # plt.show()
                    plt.savefig('FigOut/AreaComparison.png', format='png', dpi=500)
                    print '> Comparison plot saved to \"FigOut/AreaComparison.png\".'

            except ValueError:
                pass

    if print_comparison == 1:           # save comparison table to file
        with open('DataOut/ComparisonTable.csv', 'w') as f:
            f.write(comp_str)
        print '> Comparison table saved to \"DataOut/ComparisonTable.csv\".'
    if print_probabilities == 1:        # save probability table to file
        with open('DataOut/ProbabilityTable.csv', 'w') as f:
            f.write(prob_str)
        print '> Probability table saved to \"DataOut/ProbabilityTable.csv\".'

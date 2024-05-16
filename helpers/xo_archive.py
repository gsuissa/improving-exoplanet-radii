from requests import request
from collections import defaultdict
import pandas as pd
import numpy as np
from astropy.time import Time

BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+"


def get_exoplanet_parameters(
    search_name,
    which="default",
    custom_cond=None,
    columns=[
        "pl_name",
        "pl_letter",
        "pl_orbper",
        "pl_orbincl",
        "pl_orbeccen",
        "pl_trandep",
        "pl_tranmid",
        "pl_trandur",
        "pl_ratror",
        "pl_imppar",
        "st_rad",
        "st_mass",
        "st_dens",
        "gaia_id",
        "pl_rade",
    ],
):
    """Get parameters for exoplanets from the Exoplanet Archive using their TAP service

    See here for more information: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html

    Parameters
    ----------
    search_name : `str`
        Substring against which to match the planet name, e.g. "Kepler-5 " <- the space is important so you
        just get Kepler-5 and not Kepler-50 or Kepler-503 etc. Leave as `None` to ignore name conditions.
    which : `str`, optional
        Which table of parameters to draw from, one of ["default", "all", "composite"]. The default parameters
        are from a single paper that is currently flagged as the main reference. "all" will give you many rows
        for each planet, one row per publication per planet. "composite" combines the known parameters from
        many papers into a single row - NOTE composite values may not be self-consistent. By default "default"
    custom_cond : `str`, optional
        Custom condition against which to match, by default None
    columns : `list`, optional
        Which columns to select from the tables. See here:
        https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html for a list of potential choices.
        Note the columns are different for the "all" and "composite".
        By default ["pl_name", "pl_letter", "pl_orbper", "pl_orbincl", "pl_orbeccen", "pl_trandep",
                    "pl_tranmid", "pl_trandur", "pl_ratror", "pl_imppar", "st_dens", "pl_rade"]
        Default units are:
        "pl_orbper" (days), "pl_orbincl" (degrees), "pl_orbeccen" (degrees), "pl_transdep" (%),
        "pl_tranmid" (Julian date), "pl_trandur" (hours), "pl_ratro" (unitless), "pl_imppar" (unitless),
        "st_rad" (Solar radii), "st_mass" (Solar mass), "st_dens" (g/cm^3), "pl_rade" (Earth radii)


    Returns
    -------
    parameters : `list`
        A list of dictionaries, each corresponding to a row from the table

    Examples
    --------
    A query for the composite parameters of the Kepler-444 system::
        get_exoplanet_parameters("Kepler-444 ", which="composite")

    A query for the default parameters of the Kepler-29 system::
        get_exoplanet_parameters("Kepler-29 ")

    A query for the default parameters of just Kepler-9b::
        get_exoplanet_parameters("Kepler-9 ", custom_cond="pl_letter='b'")
    """
    # decide which table to pull from
    table = "ps" if which in ["default", "all"] else "pscomppars"
    from_table = f" from {table} where "

    # specify conditions on the planet name and whether to get the default parameters
    name_cond = (
        f"lower(pl_name)+like+'%{search_name.lower()}%'"
        if search_name is not None
        else ""
    )
    default_cond = "default_flag=1" if which == "default" else ""
    if search_name is not None and which == "default":
        default_cond = " and " + default_cond

    # add a custom condition if desired
    if custom_cond is not None:
        default_cond += f" and {custom_cond}"

    # force the format to be JSON
    fmt = "&format=JSON"

    # combine into URL and perform the request
    url = BASE_URL + ','.join(columns) + from_table + name_cond + default_cond + fmt

    r = request(method="GET", url=url)

    # print out an error if it failed, otherwise return the JSON
    if r.status_code != 200:
        print(f"Request failed with code {r.status_code}: Error message follows")
        print("=======================================================")
        print(r.text.rstrip())
        print("=======================================================")
        print(f"URL used: {url}")

    else:
        parameters = r.json()

        # convert gaia id strings to ints
        gaia_ids = [
            int(p["gaia_id"].split(" ")[-1]) if p["gaia_id"] is not None else np.nan
            for p in parameters
        ]

        # pass them to helper function and save Berger densities
        densities = get_berger_density(gaia_ids=gaia_ids)
        for i in range(len(parameters)):
            parameters[i]["berger_dens"] = densities[i]
            # parameters[i]["pl_tranmid_bkjd"] = Time(parameters[i]["pl_tranmid"],format="jd").bkjd
        return parameters


def get_berger_density(gaia_ids):
    """Get the stellar densities estimated by Berger+2023 associated with a collection of Gaia IDs

    Parameters
    ----------
    gaia_ids : `list`
        IDs for Gaia in DR3

    Returns
    -------
    densities : `list`
        Densities in g/cm^3

    masses: `list`
        Masses in solar units
    """
    # yes, this is such a hack please don't judge me
    base = __file__.replace("/xo_archive.py", "/")
    stellar_df = pd.read_csv(base + "GKTHCatalog_Table4.csv")
    gaia_df = pd.read_csv(base + "GKTHCatalog_Table2.csv")

    densities = np.repeat(-1, len(gaia_ids)).astype(float)
    masses = np.repeat(-1, len(gaia_ids)).astype(float)

    for i in range(len(gaia_ids)):
        if np.isnan(gaia_ids[i]) == False:
            in_gaia_table = gaia_df["dr3_source_id"] == gaia_ids[i]
            star_id = gaia_df[in_gaia_table]["id_starname"].values.tolist()

            if star_id != []:
                in_stellar_table = stellar_df["id_starname"] == star_id[0]
                density = stellar_df[in_stellar_table]["iso_rho"].values.tolist()
                if len(density) > 1:
                    print('error! gaia id found to have more than one berger density')
                densities[i] = density[0]

                mass = stellar_df[in_stellar_table]["iso_mass"].values.tolist()
                if len(mass) > 1:
                    print('error! gaia id found to have more than one berger mass')
                masses[i] = mass[0]

            # else:
            #    print('not in table!')

            if len(star_id) > 1:
                print('error! gaia id found to have more than one stellar id ')
        else:
            print('no gaia id found!')
    return densities


def transpose_parameters(parameters):
    """Transform a list of dictionaries of parameters into a dictionary of lists

    Parameters
    ----------
    parameters : `list`
        List of dictionaries of parameters

    Returns
    -------
    transposed : `dict`
        Dictionary of columns, each of which are a list of values (one for each planet)
    """
    transposed = defaultdict(list)
    keys = parameters[0].keys()
    for i in range(len(parameters)):
        for key in keys:
            transposed[key].append(parameters[i][key])
    return dict(transposed)

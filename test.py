from scipy.spatial import distance
import numpy


def steuern(einkommen):
    """BLABLABLA TESTFUNKTION"""
    if einkommen <= 50:
        steuer = 0
    elif einkommen == 51:
        steuer = 1
    else:
        steuer = 2 * (einkommen-51)

    while steuer <= 10:
            steuer += 1

    return steuer



print(steuern(input("Steuern: ")))
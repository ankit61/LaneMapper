#include"PlaneTLinkage.h"
#include"UniformSampler.h"

//TODO: Use input from XML.  This is a very crude file and is meant for only testing purposes

int main() {
	PlaneTLinkage t;
	ArrayXXf data(3, 754), samples, hypotheses, residuals, pref;
	ArrayXf clusters;

	data << -2.4836,-2.4831,-2.2932,-2.226,-2.0274,-1.9712,-1.9572,-0.55226,-0.49278,-0.2905,-0.53078,0.31472,0.56332,0.70909,0.73473,0.70442,0.8038,1.0285,1.4787,1.4678,1.4856,1.4776,1.4864,1.8149,1.9105,-2.5955,-2.4601,-2.4679,-2.327,-1.9852,-2.0252,-1.7118,-1.666,-1.6047,-1.5991,-1.4658,-1.3404,-1.1084,-1.1839,-0.99647,-0.99988,-0.78544,-0.47628,-0.62104,-0.22206,0.052551,0.15991,0.20991,-0.20111,0.5287,0.39007,0.474,-0.058696,-0.02915,0.17862,0.96763,0.22805,1.2562,1.349,0.46556,1.3768,1.4814,1.6888,1.7422,0.80096,1.8309,1.7993,0.83878,1.8044,0.8916,0.96271,1.8288,2.0097,1.8828,0.92927,2.0013,1.0799,2.1144,-2.6222,-2.3613,-2.2722,-2.1545,-2.1479,-2.0352,-1.759,-1.6397,-1.5839,-1.3563,-1.2455,-1.2039,-1.2268,-1.1416,-1.127,-1.0769,-1.0371,-1.0393,-0.91532,-0.90448,-0.96614,-0.70373,-0.6386,-0.56121,-0.18194,-0.43107,-0.35229,0.098277,0.22427,0.16795,-0.24225,0.2576,0.48193,0.7785,0.86366,0.2627,1.168,0.37524,1.5729,1.6198,1.6227,1.4745,1.6752,1.5754,0.85187,0.85485,0.90857,0.81586,0.99418,1.9087,0.93868,1.0125,1.0109,2.2786,-2.406,-2.1634,-1.9304,-1.5739,-1.2622,-1.2383,-1.0315,-0.85625,-0.83154,-0.46617,0.0063134,0.55795,0.65957,0.33892,0.37617,0.41928,1.1095,1.1265,1.467,0.48995,1.3611,1.4688,0.79879,1.5689,1.6302,0.79935,0.91957,1.9314,1.9609,-2.4553,-2.4836,-2.5761,-2.5348,-2.4471,-2.3515,-2.0033,-2.1043,-1.9747,-2.033,-1.9376,-1.9318,-1.9194,-1.7349,-1.7326,-1.6939,-1.6222,-1.564,-1.4367,-1.166,-1.1573,-0.99032,-0.69581,-0.49923,-0.31138,-0.4322,0.12615,-0.2441,0.29332,0.69247,0.65482,0.11315,0.16553,0.20462,0.19494,0.21715,1.0279,0.27128,0.35091,0.33716,0.39307,1.1924,1.2275,1.1369,1.2407,1.1975,0.4602,0.48649,1.3311,1.2915,0.5422,1.3623,1.4907,1.6014,1.4675,1.6616,1.6574,1.6616,1.7056,1.844,1.7286,1.9387,1.9101,1.8659,2.0596,1.9395,1.0639,1.9746,2.0164,1.1123,2.0071,2.1925,2.1893,2.3042,-2.4451,-2.3526,-2.2736,-2.1903,-1.8702,-1.9378,-1.9297,-1.9259,-1.7531,-1.7398,-1.6774,-1.6706,-1.6681,-1.665,-1.4366,-1.0942,-1.094,-1.0111,-0.97387,-0.7965,-0.39726,-0.53586,-0.2909,0.27875,0.55916,-0.078667,0.054315,0.1436,0.8774,0.23679,1.0998,0.39869,1.2257,0.38399,1.2733,1.1751,1.3767,1.4046,1.5698,1.7351,1.8869,1.8331,0.92887,2.1258,2.2883,-2.5147,-2.4679,-2.0742,-2.144,-1.8776,-1.8752,-1.8527,-1.7394,-1.7732,-1.7256,-1.6359,-1.4157,-1.4193,-1.3655,-1.2524,-1.0698,-1.0778,-1.0704,-1.0284,-1.0329,-0.73493,-0.23579,-0.2289,0.0084415,-0.052963,0.26244,0.13393,1.0234,0.91368,0.33406,1.1037,1.2393,1.4673,1.6236,1.6603,1.5992,0.83777,0.79038,0.88764,1.7249,-2.4796,-2.6059,-2.4131,-2.3133,-1.9864,-1.9605,-1.9088,-1.7389,-1.7129,-1.6046,-1.5444,-1.4541,-1.4128,-1.1926,-1.1237,-1.0551,-0.55229,-0.4301,-0.6867,-0.086731,0.32458,0.32686,0.13298,0.1518,1.1322,1.3648,1.3725,1.1727,1.5595,0.71192,1.8464,2.0068,1.897,1.9433,2.0433,2.1006,1.9957,2.0939,2.1365,-2.4326,-0.56331,0.15911,1.1729,-2.3416,-1.9321,-1.7569,-1.6618,-1.6231,-1.4846,-1.1172,-0.8693,-0.69641,-0.70518,-0.51195,0.24737,0.62737,1.218,1.2536,1.4326,1.8047,0.85797,2.2268,2.0418,-1.2297,-1.0953,-0.90079,0.5707,-0.066066,0.93894,1.2651,1.3359,1.858,1.892,1.97,2.0298,2.1691,2.0871,-0.93114,-0.42677,1.9353,0.02677,0.51939,1.4333,2.2168,1.6446,1.629,2.4094,-1.6423,-1.2123,-1.065,-0.76389,-0.17233,1.0298,1.5339,1.1938,1.2506,1.7934,1.6984,1.7009,2.4528,1.8506,-1.7661,-1.512,-1.4466,-1.4183,-1.0965,-1.0551,-0.8277,-0.36345,-0.19381,0.3046,0.538,1.0213,1.2121,1.0855,1.1132,1.3921,1.4075,1.5278,1.4844,1.5429,1.5667,1.5989,1.7671,-2.3515,-1.6363,-1.5013,-1.4974,-1.4974,-1.35,-0.38283,1.6343,2.4106,2.4626,1.7857,-2.5483,-2.2003,-1.6223,-1.5828,-1.3822,-1.1091,-0.6633,0.37394,0.36384,0.39278,0.5627,0.66957,1.0684,1.3528,1.4103,1.4226,2.0505,1.5483,2.3225,2.2646,2.4284,1.7935,2.5925,1.8584,1.8789,-2.2319,-1.9781,-1.971,-1.769,-1.6527,-1.5287,-0.54857,-0.20696,0.38578,0.77601,0.85898,0.77218,1.4066,1.58,1.2803,1.3223,1.4053,2.1119,2.3805,2.2858,1.7188,2.3782,2.3938,2.3736,2.4309,1.7994,1.8521,-1.9586,-1.7639,-1.7962,-1.3619,-0.96086,-0.56963,-0.18506,-0.16081,-0.13987,0.027525,1.0214,1.0535,1.4217,1.1371,1.174,1.2415,1.3871,1.4269,1.5081,2.1596,2.1551,1.6156,2.4939,-0.90554,-0.57343,-0.058578,0.3334,1.0218,1.2041,2.0268,2.2647,2.4231,0.91274,2.4689,-1.2009,-0.35522,0.44133,0.76023,1.2126,1.3468,2.232,2.2892,-1.063,-0.60012,-0.42084,0.49333,0.55713,1.2764,2.143,2.2476,-1.8997,-1.7593,-1.6618,-0.50955,-1.2093,1.1986,-1.5258,0.10554,1.7151,-2.0815,-1.6683,-1.5377,-1.4615,-0.37514,0.81234,0.94764,0.26917,0.11709,-0.82426,2.0126,-1.6839,-1.6473,-1.0798,-2.138,-0.99692,0.33821,-2.3628,-2.0458,-2.0147,-1.6373,-1.5508,-1.1568,-0.93433,-1.8821,-0.93629,1.2775,-1.7665,-0.59893,-0.13782,-2.3325,-2.2879,0.75397,1.2434,2.0066,-2.3772,-2.3032,-1.7104,-1.6266,-1.0133,-0.68343,0.13525,0.73779,0.99545,2.1501,-2.2416,-1.9586,-1.7953,0.94609,-2.5114,-1.9583,-0.99145,-0.81818,-2.0141,-1.436,-0.87716,-0.72703,-2.4814,-1.7531,-1.3631,-0.59051,1.1822,-1.9417,-1.5159,1.8968,1.0217,-1.9331,-0.64267,0.3215,-2.2051,-2.2326,-1.6265,-2.3459,0.68628,-0.33945,0.87147,-1.0106,-1.6926,-2.0199,-2.3457,-1.5028,-1.1696,-0.95877,-0.90976,-0.90777,-0.87693,-0.77437,-0.47351,0.70286,1.2788,-1.6625,-0.3885,0.58002,0.12695,-1.9687,-0.63862,-1.7161,-0.91414,-1.3256,-1.4586,0.56389,0.63881,1.8961,2.129,2.1351,2.2143,-2.2973,-1.5372,-1.3362,-1.0034,-0.57221,0.3565,0.33807,0.53715,1.6102,1.7536,2.6423,2.7052,2.6593,2.3242,2.8514,-0.59461,2.4544,-2.4536,-1.4782,-0.58513,-0.50295,-0.49053,-0.21533,2.2698,2.7531,2.2637,2.8806,-2.088,-2.0765,-0.86456,0.29761,0.12459,0.18825,1.0006,1.156,1.7581,1.4592,1.5459,1.5661,1.8614,1.8684,2.6585,2.178,2.3555,2.881,-2.0531,-0.90164,-0.54483,-0.18722,0.66564,0.73407,2.4213,2.1788,-1.4737,-0.844,0.3668,0.37517,1.8997,1.886,-0.37512,0.10712,0.43862,2.0881,-2.1926,-1.8819,-0.82021,-0.6991,0.21074,0.97938,1.5152,1.8797,1.8912,-2.376,-2.0391,0.082981,0.57746,2.0743,2.1787,2.8211,-1.5821,-0.54479,-0.13805,
-1.4734,-2.2123,-1.4257,-1.4487,-1.517,-2.096,-1.9831,-1.4691,-1.3137,-1.1233,0.92487,0.61796,-0.89316,-0.84333,-0.033482,0.25139,0.4485,-0.44066,-1.5041,-0.53274,-0.67267,0.31316,0.53023,-1.0224,0.3511,-2.159,-1.8635,-2.2336,0.23722,1.153,-2.0267,1.2872,1.3547,-1.3471,-1.6845,-0.085766,0.18272,-1.8801,1.3758,-1.4691,-0.59128,-0.72926,-1.2135,0.76069,-1.444,-1.38,0.24209,-0.14694,1.4171,-1.316,0.2813,-0.083845,1.288,1.4456,0.87617,-1.0001,1.0007,-0.10043,-0.68238,1.2609,0.56147,0.5787,-0.42881,-0.71416,0.96265,-0.90897,-0.52146,1.0121,-0.36101,1.0353,0.77586,0.10667,-0.82778,0.27883,1.4219,0.32189,1.4099,0.24206,-2.1234,1.2392,-1.4937,-2.1249,-2.0078,-1.7542,1.3518,-1.6167,-1.9554,-1.4252,-1.9449,-1.2026,1.0366,-1.8885,1.3618,-0.11806,-0.3221,1.0652,-1.5221,-1.4891,1.052,-0.64983,-0.39129,1.0037,-0.56172,1.097,1.3589,-0.51951,-1.1481,0.20358,1.4286,0.046723,-0.6083,0.15907,-0.27288,1.3799,-0.64793,1.4122,-0.68145,-0.93175,-0.447,0.65185,-0.18336,0.68468,0.87203,0.97844,0.84927,1.4656,0.7695,-0.0261,1.3932,0.99447,1.0723,-0.95327,0.37554,-2.0613,-1.919,0.40626,-1.482,-0.33595,-1.3216,-0.61131,1.3308,0.93175,-0.40837,0.45307,0.066468,0.85966,0.90269,0.99008,0.3975,0.47989,-1.3607,1.3584,0.34675,0.33226,0.63507,0.71056,0.56312,1.4039,1.3099,0.64493,0.63842,0.25065,-0.26265,-2.135,-2.2352,-1.8835,-2.2552,0.95237,-1.886,1.059,-2.0469,-1.9931,-2.1022,-1.7309,-0.86671,-1.9953,-0.050917,1.0474,-1.7545,0.26101,-1.8742,-1.9148,0.44771,-0.42062,-1.288,-1.3025,1.364,0.20586,1.3073,-0.68305,-1.1426,0.10363,1.076,0.89075,0.82176,0.99493,0.94483,-0.79195,0.9588,0.90447,1.1118,0.88543,-0.63196,-0.60512,0.43943,-0.076205,0.38638,1.2955,1.2924,-0.051044,0.36632,1.2281,0.25979,-0.49795,-0.43572,0.63075,-0.4005,-0.19688,-0.16686,-0.31851,-0.2478,0.57515,-0.015783,0.23457,0.5612,-0.60486,0.20074,0.87775,0.19547,0.013864,0.87163,0.18596,-0.83087,-0.22768,-0.78316,-2.118,-1.9474,-1.4203,-0.4068,1.0769,-1.9684,-2.0306,-1.8173,1.2058,-2.1155,-1.8004,-0.070711,-1.7161,-1.3006,-1.384,-1.7935,1.3412,1.2055,0.93774,-0.75933,-1.316,1.2146,0.055522,0.12738,-1.2575,1.3057,0.80352,0.83659,-0.77192,0.86251,-0.47776,1.0183,-0.65876,1.254,-0.72354,0.7447,-0.16409,0.24755,-0.42962,-0.51199,-0.65607,0.54281,1.4394,-0.54252,-0.97927,-0.40421,-2.102,0.67337,-1.541,-1.4885,-1.602,-1.4223,1.0473,-1.4747,-0.43358,1.0012,-1.416,-0.38262,-0.77849,-0.12749,-1.8185,-0.73065,1.0054,0.077065,1.1538,-1.2468,-1.2654,0.00011579,-1.4017,0.53899,-1.2672,1.2476,-0.70396,0.3245,1.034,0.43545,0.41904,0.39427,-0.41402,-0.57023,-0.045887,0.79036,1.3853,0.91777,0.54838,0.20239,-2.1828,-2.1439,-0.92174,0.30287,-1.656,-0.072714,-0.83453,-0.25273,-1.4933,-0.16342,-2.0369,0.38632,0.99369,-1.3722,0.96978,-0.49809,-1.1398,1.3681,-0.60381,-0.73949,0.092023,0.80903,0.91518,0.023853,-1.2312,-1.1865,0.4341,-0.16523,1.3485,-0.21588,-0.80198,0.14265,0.23424,-0.33994,-0.62926,0.24055,-0.34711,-0.42432,-1.5809,1.3348,1.0672,0.28707,-0.90511,-1.8678,1.0971,1.1606,-1.6516,-1.9097,-1.8497,-1.3315,-1.4129,-0.27912,0.89783,0.23006,-1.0909,0.45787,0.38052,0.36802,-0.39218,1.0164,-0.90349,0.33199,1.122,-0.77587,-0.32424,-0.82767,1.3385,-1.1081,-0.88537,-0.73159,-0.3353,-0.23491,-0.49393,-0.38379,-0.77589,-0.11172,-1.0775,0.93879,0.11009,0.18611,-0.71366,-1.4676,-0.63093,0.81297,1.4388,0.30267,-1.2725,-1.995,-2.1592,-1.5248,-0.73741,0.7325,-0.11202,1.5929,1.5139,0.64787,1.1994,1.3482,0.11186,1.5836,-0.1299,-0.20972,0.28119,-1.4978,-1.1158,-1.3408,-0.83408,0.90814,1.6147,0.50452,0.42786,0.13249,-1.4499,1.4437,1.4139,1.5019,1.3789,0.78387,1.4481,0.97185,0.76354,0.91692,0.80386,-2.2113,-0.65963,-1.5397,-1.9585,-2.0706,-1.5914,-0.22795,0.79934,0.10276,0.089659,1.494,-2.1303,1.0031,-0.14836,-1.5989,-0.8079,-1.4062,-0.41287,-0.74086,0.361,1.5638,1.6242,1.5127,1.5717,1.2444,1.3937,1.3054,-0.24904,0.80684,-0.53256,0.20671,0.21591,1.5988,-0.63399,1.3881,1.4094,-0.38012,0.33031,-1.4258,-0.39535,-0.50698,-1.6638,1.5589,0.67143,0.6007,-0.93726,-1.0995,1.2198,-1.2783,-0.8387,1.4009,1.4346,1.467,0.45533,-0.81456,0.34628,0.88951,-0.49115,-0.51447,0.049132,-0.45934,1.3516,1.3524,-0.72043,0.73869,-0.22992,0.81276,-1.2836,-0.99863,-0.93789,-0.19799,0.099596,1.6631,0.13262,0.5995,-0.17131,1.3279,1.1523,1.2359,1.4181,1.4096,0.71876,0.087061,0.36736,0.78287,0.49724,0.37222,1.5292,1.5948,1.6188,0.66362,1.2102,0.54408,-0.38937,0.46156,1.4644,-0.41561,1.4334,1.384,1.6089,1.5804,1.2521,1.2759,-0.42439,-0.43353,-0.36745,1.3165,1.1722,0.28132,1.4261,1.449,0.043359,0.28432,-0.022599,-0.47532,-2.0786,1.3292,-1.3377,-0.609,-1.7223,-1.0547,-1.2834,0.3875,-1.8315,-1.6359,0.17764,-0.1005,1.0439,0.82055,-1.3791,0.89596,0.82231,0.26306,-0.32385,-0.68726,-1.3653,0.64296,0.47181,0.83582,0.43659,0.23817,-0.87471,-0.26701,-0.57422,-0.5017,-0.35676,-1.5643,-0.52162,0.27468,-0.86191,-0.42152,-0.58334,-1.565,-1.6353,-1.1681,-0.17148,-0.70847,-2.1778,-1.4988,-0.05435,-0.69637,-1.345,-0.36485,0.68833,-0.40039,-0.64993,-0.72958,-0.41183,-1.4994,0.67847,0.99993,-2.1978,-1.4725,-1.9899,-1.3387,-0.90881,-0.40337,-0.32071,-0.93161,-2.075,-0.45373,0.02486,0.24146,-0.70881,-1.5463,-1.94,0.070095,0.93603,-1.4322,-0.77809,0.94015,0.9472,-0.45091,-0.63769,-1.2964,-0.82352,1.3293,1.072,0.5148,0.26006,-0.68825,-1.2964,-1.9382,-1.6863,-1.1599,-0.54035,-1.0188,-0.68214,-0.61383,0.2573,-0.39406,1.4843,-0.48314,1.3617,0.10024,1.3167,1.4681,-0.77393,-0.10557,-1.3747,1.3066,-0.5689,-1.2892,-0.7607,0.41738,1.1872,1.3033,1.0738,0.23019,-1.6367,0.025541,0.47393,1.5674,1.7666,0.079923,1.52,1.252,0.90309,0.19716,-0.38872,0.55761,0.76578,-0.56288,-0.80928,0.46682,-2.0747,-2.0691,-0.4933,-0.26636,-0.78806,-0.87646,0.68482,-0.43279,1.1367,-0.60504,1.831,1.87,1.5079,1.5105,0.12021,-0.054452,1.711,1.697,-1.2308,1.651,1.0216,0.83531,1.4109,1.5691,-0.51012,1.1329,0.6521,-0.55372,0.23466,-0.68549,1.4878,-0.83543,1.4855,-1.2659,-0.48896,1.2925,0.42898,1.7043,1.4684,1.1185,1.238,1.4698,1.6648,1.5991,0.068386,0.37228,-0.21791,-1.5663,1.6678,0.32745,0.86889,1.6105,-0.17708,1.4418,1.3276,-1.4513,0.63868,-1.2419,1.5675,0.54233,1.0662,-0.59066,-1.5382,1.5815,-0.58928,
7.6707,7.9928,7.7574,7.8239,7.9826,8.2789,8.2685,9.0809,8.9306,8.9877,6.639,8.6707,9.454,9.524,9.2119,9.1033,9.063,9.2927,9.3996,9.103,9.1183,8.7703,8.6838,9.0564,8.4822,7.8808,7.8435,8.0199,7.0449,6.8263,8.2286,6.7224,6.7175,8.1939,8.3527,7.7727,7.7202,8.7494,6.5985,8.6457,8.3092,8.4953,8.8906,6.7014,9.148,9.3068,8.7546,8.9115,6.3979,9.5945,8.854,9.0557,6.415,6.362,6.5128,9.5071,6.4634,9.0544,9.1894,6.3378,8.7425,8.6877,8.9262,8.9853,6.3684,9.0209,8.916,6.339,8.8469,6.3246,6.3913,8.664,8.8967,8.5758,6.1782,8.4955,6.1564,8.4765,7.8699,6.5979,7.8093,8.1735,8.1422,8.0901,6.7121,8.2946,8.4977,8.3813,8.705,8.4112,6.7202,8.745,6.5957,8.0491,8.1578,6.6994,8.7235,8.7192,6.6651,8.5249,8.4446,6.6173,8.8359,6.5571,6.4621,8.9958,9.3407,8.7624,6.4017,8.8738,9.2902,9.163,9.3091,6.3256,9.2895,6.2944,9.0946,9.1357,8.9679,8.6591,8.8469,8.5722,6.381,6.3446,6.379,6.201,6.3936,8.6543,6.1843,6.3055,6.2916,8.7984,6.9472,8.1474,8.2458,7.5153,8.4888,8.0343,8.5748,8.3996,6.5546,6.6096,8.8958,8.9132,9.1364,6.4906,6.4761,6.4272,8.9473,8.8947,9.3444,6.2886,8.8224,8.7672,6.471,8.577,8.607,6.2122,6.2181,8.4151,8.3987,6.9547,7.1549,7.8745,7.9468,7.8698,8.0622,6.9008,8.1268,6.8625,8.2234,8.2873,8.31,8.1585,7.9148,8.407,7.6306,6.7977,8.4057,7.6305,8.7572,8.7535,7.8598,8.4357,8.9181,9.0389,6.4572,8.7537,6.4508,9.1842,9.6265,9.1192,6.4674,6.5117,6.5217,6.4681,6.4814,9.4077,6.471,6.4704,6.4026,6.4682,9.2962,9.3015,8.9218,9.0475,8.9023,6.3178,6.3069,8.9881,8.8569,6.315,8.8509,9.0583,8.9737,8.6675,8.9097,8.8761,8.861,8.8835,8.7784,8.5453,8.6372,8.5772,8.4859,8.7909,8.5658,6.3352,8.5478,8.5935,6.3349,8.5313,8.7972,8.5836,8.7346,7.958,7.9566,7.7802,7.4111,6.8456,8.269,8.2828,8.2066,6.7654,8.4435,8.358,7.658,8.3223,8.1559,8.3289,8.7385,6.6015,6.6258,6.7139,8.5099,8.9658,6.5322,8.5033,8.8517,9.5993,6.4214,6.5575,6.5271,9.4839,6.5062,9.2509,6.4199,9.2841,6.3454,9.2492,8.7846,8.9954,8.8405,8.9969,8.9295,8.9003,8.5079,6.1817,8.715,8.7892,7.1865,7.9444,7.0496,7.9301,8.0768,8.1437,8.0612,6.8227,8.1545,7.7497,6.8115,8.3662,7.9353,8.1276,7.9369,8.7727,8.3104,6.7102,8.0098,6.658,8.7487,9.0795,8.5636,9.2874,8.4697,9.4007,6.3956,9.3815,9.0719,6.4332,8.9328,8.8758,8.7489,8.9475,8.9974,8.8493,6.4084,6.2235,6.3704,8.5597,6.9662,7.8834,7.988,7.5607,7.2623,8.1024,7.472,7.9064,7.6649,8.2779,7.7716,8.5865,7.6108,6.7433,8.5402,6.7101,8.5666,8.907,6.5213,8.9083,9.2255,8.9071,6.5396,6.5083,9.0601,9.3745,9.3641,8.8958,8.9143,6.2411,8.7729,8.9021,8.6283,8.5607,8.7004,8.7568,8.5263,8.6579,8.6839,7.7533,6.5109,6.4597,8.9583,7.5184,8.2183,6.8116,6.7723,8.3178,8.5405,8.7478,8.6976,8.8338,8.3602,6.6352,8.7846,9.555,8.8728,8.8691,8.7841,8.8693,6.3298,8.7969,8.463,6.6931,8.3188,8.2646,9.4396,6.4035,9.5494,9.3112,9.2158,8.8118,8.7398,8.8033,8.7327,8.7924,8.6152,8.5415,6.6169,8.61,8.6652,9.3475,9.4006,8.703,6.2352,6.0317,8.2646,8.149,8.7347,8.7187,8.8502,8.9069,8.8764,8.9019,6.0657,6.0812,8.4839,6.0953,6.0457,8.3135,5.9272,7.5951,7.7957,7.6333,8.3979,8.4312,8.5578,8.5087,6.6092,6.3554,8.7128,8.9113,9.0895,9.5225,6.1412,6.1469,6.0536,6.0961,6.2728,6.0563,6.202,6.2756,6.2085,6.2148,8.04,7.8821,8.3495,8.5314,8.5755,8.4736,8.5577,6.246,8.3379,8.3155,5.9755,7.8669,6.8362,7.7333,8.3195,8.1233,8.5569,8.4562,9.2648,8.8248,6.2539,6.1901,6.2016,6.0985,6.15,6.0855,6.1086,8.6626,6.2575,8.6051,8.3777,8.2845,5.9343,8.4904,5.9987,5.9815,7.3818,7.285,7.9698,7.6824,7.8368,8.3757,6.4401,8.3056,8.7491,9.5965,9.6034,6.2796,9.3593,9.1123,6.1139,6.0998,6.0645,8.3848,8.6636,8.3242,6.1981,8.5606,8.5598,8.3712,8.5174,6.0233,6.0101,7.6947,7.2379,7.614,7.481,8.6013,8.757,8.9988,8.7092,8.5991,6.2826,9.0894,8.8974,8.9773,6.1565,6.2253,6.1775,6.0849,6.0745,6.2982,8.4762,8.3798,6.2537,8.148,7.9548,6.4397,6.3354,6.2402,8.8939,6.1855,8.3955,8.5914,8.1986,6.1682,8.4721,6.6065,6.4545,6.2224,6.1673,6.1708,6.139,8.6183,8.5876,8.1566,6.5294,6.5385,8.9522,6.2545,6.0969,8.5072,8.3607,7.4496,7.742,8.4804,6.4947,8.4634,9.3125,8.4175,9.2225,9.1869,7.1633,8.3756,8.3733,7.6646,8.5153,6.3334,6.3791,9.4399,6.528,7.8252,8.5213,7.7151,7.8976,8.5539,7.0097,7.8684,6.4919,6.945,7.236,7.7406,7.7274,7.9215,8.1481,8.2483,8.1066,8.3054,8.8953,7.8883,8.4993,8.8679,7.8062,7.8653,9.6752,9.0657,8.8371,8.0223,7.79,7.6046,7.9133,8.6083,8.4217,8.5251,9.3749,9.3791,8.7825,7.3809,8.0317,7.2206,6.321,7.9234,8.0264,8.8743,8.7314,7.7428,7.9131,8.2662,8.6378,7.9207,7.727,7.7896,8.2274,9.2878,8.0614,8.534,8.6374,6.3263,8.0138,8.6144,6.4619,6.8406,7.4101,7.8947,7.6857,9.4972,6.4636,6.3106,7.8576,7.4793,7.6452,7.6855,8.5241,8.6336,8.5612,8.3463,8.533,8.4215,8.4703,8.2983,9.3453,6.0842,7.8085,6.4704,9.0795,6.3784,6.7457,8.6075,7.6282,8.6705,6.6686,7.9779,9.5887,9.4517,8.5118,6.0054,5.9702,6.028,7.0767,8.3668,7.824,7.8783,6.4266,6.1919,8.9285,6.2236,6.094,6.1786,8.1724,8.351,8.0416,6.1224,8.3288,8.6675,8.1785,7.9239,8.6009,8.539,8.4976,8.7118,8.9399,6.1547,8.3321,5.9931,8.3117,6.565,6.5554,6.5157,6.2833,8.7573,8.8889,6.0698,6.0427,9.1443,5.9857,6.1868,6.2434,5.9943,5.9303,8.4003,6.0238,6.147,8.3061,7.2454,8.408,6.4587,8.9506,6.2083,9.6974,8.5311,5.9645,7.5527,6.4592,6.278,6.3916,6.041,5.9655,6.3729,6.2947,8.9946,8.41,7.3319,8.1052,6.4646,8.1312,6.5124,6.1026,8.9257,5.9794,6.0097,7.7131,7.0787,9.2551,6.2105,8.3709,6.0369,8.3467,8.2991,6.4361,8.8677;


	t.SetSampler(TLinkage::UNIFORM);
	t.Sample(data, samples, 5 * data.cols());
//	cout << endl << endl << samples << endl << endl;
	t.GenerateHypothesis(data, samples, hypotheses);
//	cout << endl << endl << hypotheses << endl << endl;
	t.FindResiduals(data, hypotheses, residuals);
//	cout << endl << endl << residuals << endl << endl;
	t.FindPreferences(residuals, pref, TLinkage::EXP, 0.13);
//	cout << endl << endl << pref << endl << endl;
	t.Cluster(pref, clusters);
//	cout << endl << endl << clusters << endl << endl;
	t.RejectOutliers(clusters, clusters);
//	cout << clusters << endl;
	vector<ArrayXf> models;
	t.FitModels(data, clusters, models);
//	for(int i = 0; i < models.size(); i++)
//		cout << models[i] << endl << endl;
}

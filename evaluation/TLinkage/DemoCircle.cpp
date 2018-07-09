#include"Circle2DTLinkage.h"
#include"UniformSampler.h"

//TODO: Use input from XML.  This is a very crude file and is meant for only testing purposes

int main() {
	Circle2DTLinkage t;
	ArrayXXf data(2, 500), samples, hypotheses, residuals, pref;
	ArrayXf clusters;

	data << 0.72766,0.71606,0.71309,0.6764,0.65045,0.61911,0.57772,0.52625,0.46485,0.41538,0.34593,0.26042,0.20638,0.13016,0.071057,-0.0061976,-0.074753,-0.13196,-0.17451,-0.23888,-0.29862,-0.32945,-0.34964,-0.37019,-0.37463,-0.3718,-0.36517,-0.35448,-0.31845,-0.27487,-0.24163,-0.19855,-0.13422,-0.077305,-0.0095522,0.059392,0.10905,0.20122,0.26874,0.33055,0.40543,0.46607,0.52526,0.56258,0.61384,0.65215,0.68018,0.70878,0.72809,0.73627,0.91365,0.89433,0.89988,0.88948,0.8606,0.84366,0.79896,0.78513,0.72978,0.69323,0.65986,0.62195,0.59554,0.54304,0.49316,0.45992,0.41449,0.37218,0.34029,0.31197,0.29167,0.26848,0.25399,0.24424,0.21778,0.2413,0.23864,0.25295,0.27505,0.29132,0.31293,0.35744,0.38341,0.41024,0.45656,0.49311,0.55792,0.574,0.63518,0.65971,0.69994,0.73941,0.77107,0.81315,0.83513,0.86428,0.8867,0.88007,0.89006,0.91481,0.6019,0.59375,0.58738,0.58521,0.56638,0.56465,0.54029,0.5509,0.52085,0.513,0.50437,0.46474,0.44978,0.42891,0.41225,0.40364,0.36476,0.37111,0.35333,0.33616,0.3281,0.31143,0.31707,0.30304,0.29536,0.29696,0.31294,0.30747,0.31238,0.32705,0.33411,0.35075,0.36165,0.3829,0.40657,0.41149,0.4383,0.44918,0.46859,0.48923,0.50547,0.51996,0.52931,0.54595,0.57042,0.58773,0.57554,0.58719,0.60034,0.59495,0.95083,0.93838,0.92541,0.91528,0.86654,0.83414,0.77767,0.71665,0.672,0.60236,0.52967,0.45134,0.37379,0.29815,0.22878,0.14979,0.074518,0.008989,-0.033606,-0.096733,-0.15535,-0.1844,-0.21175,-0.22449,-0.227,-0.22803,-0.22896,-0.21101,-0.17692,-0.13803,-0.094555,-0.044119,0.0096449,0.074172,0.14937,0.21771,0.28976,0.37569,0.43805,0.52471,0.60289,0.66315,0.72022,0.77788,0.83142,0.87081,0.89875,0.93642,0.95235,0.94826,0.54393,0.5371,0.53578,0.50954,0.49192,0.48347,0.4581,0.42412,0.40142,0.35758,0.34174,0.30372,0.27118,0.23927,0.19019,0.15748,0.12743,0.10274,0.077613,0.04769,0.02295,0.014439,-0.010627,-0.0059651,-0.030117,-0.023017,-0.01969,-0.024927,0.00015575,0.022314,0.041282,0.075376,0.09363,0.13297,0.15774,0.18691,0.23887,0.26917,0.29419,0.3386,0.36996,0.39893,0.42669,0.46155,0.48859,0.49779,0.52654,0.52962,0.5417,0.53238,0.15605,-0.22526,0.80907,0.77452,0.32567,0.0016288,0.75366,0.43219,-0.13052,0.71421,0.86547,0.78023,-0.11327,0.75687,0.24262,0.73359,0.79159,-0.11878,0.25748,0.94682,-0.34189,-0.35002,0.3578,0.2618,0.7852,0.92528,0.52697,0.24141,0.54422,0.8135,-0.32049,-0.35023,-0.27297,0.23761,-0.13381,-0.36556,-0.11513,-0.33525,-0.0043291,-0.027275,-0.11065,0.19449,0.6134,0.095365,-0.2591,-0.083915,0.10143,-0.14989,0.50582,-0.076163,0.058432,0.1731,-0.0024948,-0.093688,0.1529,-0.29158,0.25036,0.50611,0.19081,-0.15552,0.93205,0.70619,-0.066805,-0.0062232,0.77813,0.74285,0.33711,-0.28013,0.44122,-0.20028,0.35237,0.60379,0.86965,-0.33191,-0.1103,-0.33064,-0.22877,-0.27428,0.45823,0.39778,0.84365,0.92257,0.183,0.15277,-0.30694,0.7715,0.95101,0.28795,0.45853,0.81433,0.79971,-0.25892,0.54826,0.94145,0.85624,-0.080451,-0.11974,0.90707,0.8305,0.63367,0.20345,-0.15641,0.80499,-0.34882,0.025519,0.71149,0.907,-0.029925,0.24535,-0.04362,0.92564,0.10751,0.4522,-0.071685,0.90472,-0.10115,0.33204,0.11965,0.85882,0.23806,-0.36333,-0.32583,0.035084,0.82277,0.42015,0.41476,0.90333,0.90574,0.34692,0.096766,0.77317,0.016544,0.81636,0.88804,0.24372,-0.32463,0.24088,0.15759,0.23603,0.83716,0.61345,0.15697,0.28368,0.69903,0.57179,-0.32185,0.35302,0.13554,0.11879,-0.15486,0.63999,-0.18774,0.89279,0.68274,0.69242,-0.031323,-0.2684,-0.21,0.2019,0.74166,-0.21539,-0.19226,0.11549,-0.16744,0.48781,0.27365,0.31254,0.67598,0.50797,-0.32265,0.87519,0.34814,-0.31799,-0.069018,0.68331,0.6786,0.64981,-0.22525,0.25443,-0.054321,0.55039,0.58397,0.0075929,0.18019,-0.16821,0.31321,0.6154,0.54247,0.46997,-0.19545,0.64933,-0.1283,0.34884,0.29064,0.33887,0.41763,-0.021715,0.18693,-0.32802,-0.078522,0.050314,0.31428,0.67092,0.20406,0.19825,-0.1837,0.17072,0.24108,0.23621,0.70109,0.79571,0.61155,-0.13851,0.85791,-0.14341,0.83948,-0.32456,0.77114,0.52549,-0.2465,0.17135,0.050533,-0.12908,0.84158,-0.095546,0.71632,-0.017996,0.63973,0.018245,0.16615,0.19395,-0.36792,0.88037,-0.053881,0.16609,0.16682,-0.16603,0.13105,0.76593,0.63753,0.80907,0.59941,-0.23554,0.2977,0.35656,0.075232,0.63085,0.42272,0.52616,0.63522,
	0.15357,0.21778,0.27767,0.35144,0.41931,0.47549,0.52952,0.56974,0.60347,0.64743,0.67872,0.68889,0.69934,0.69286,0.68657,0.67166,0.62505,0.58814,0.54499,0.51064,0.45214,0.37953,0.32264,0.2422,0.18099,0.12784,0.040631,-0.021015,-0.088364,-0.15413,-0.21688,-0.27352,-0.30595,-0.34554,-0.37695,-0.38287,-0.40507,-0.39991,-0.39165,-0.38136,-0.35194,-0.30999,-0.29302,-0.23939,-0.18529,-0.13352,-0.059129,0.013715,0.093355,0.15742,0.9691,1.009,1.0554,1.1005,1.129,1.1746,1.2102,1.2356,1.2601,1.2695,1.2899,1.2962,1.3098,1.3051,1.2972,1.2838,1.2688,1.2418,1.2191,1.18,1.1578,1.1117,1.0692,1.0443,0.98727,0.95496,0.90858,0.88326,0.81213,0.79395,0.75976,0.72992,0.69787,0.6745,0.66628,0.63818,0.63314,0.63761,0.64422,0.63711,0.66726,0.68684,0.70997,0.73806,0.76835,0.82015,0.85949,0.88154,0.93303,0.97465,0.5862,0.59052,0.61173,0.61954,0.64236,0.67474,0.68254,0.68948,0.70619,0.70796,0.72324,0.70988,0.7335,0.72569,0.72603,0.71136,0.69922,0.68911,0.67237,0.67427,0.64677,0.65038,0.61956,0.61498,0.58631,0.56005,0.54973,0.54243,0.50999,0.49409,0.48419,0.47128,0.45252,0.45043,0.43286,0.43171,0.43837,0.43624,0.44217,0.43988,0.44764,0.44561,0.46456,0.47855,0.48747,0.51624,0.53216,0.53694,0.54986,0.56547,0.90462,0.97335,1.0609,1.1142,1.1824,1.2613,1.3182,1.3643,1.4136,1.4555,1.4649,1.4795,1.5105,1.4868,1.4609,1.4604,1.4179,1.3844,1.3335,1.2821,1.2294,1.1615,1.082,1.0036,0.95277,0.86303,0.77397,0.71585,0.64154,0.5726,0.51756,0.45426,0.4098,0.35958,0.34186,0.31804,0.28734,0.28541,0.31737,0.32306,0.35388,0.38168,0.43308,0.47769,0.55431,0.61255,0.65951,0.75897,0.82163,0.90118,0.021923,0.039991,0.10509,0.12101,0.16015,0.19416,0.21779,0.22764,0.25445,0.27087,0.28827,0.29451,0.29738,0.31031,0.30462,0.29657,0.27381,0.24007,0.21132,0.19304,0.17539,0.13844,0.10519,0.069958,0.045258,0.0076521,-0.036391,-0.066185,-0.09159,-0.13606,-0.18455,-0.18076,-0.2339,-0.2282,-0.24469,-0.25658,-0.265,-0.25657,-0.24809,-0.24551,-0.2341,-0.22193,-0.19291,-0.17198,-0.16513,-0.11647,-0.083246,-0.056608,-0.018305,0.012848,0.20065,-0.35773,-0.056317,1.0875,0.885,0.2655,-0.23661,-0.35641,-0.12575,0.32261,0.75639,0.31009,1.3852,0.1181,0.18267,0.052766,0.31559,0.65619,1.3301,0.090438,1.1842,-0.11732,0.77318,0.16749,0.41609,0.60821,0.72215,0.083881,0.88128,0.54957,0.30435,1.233,0.48373,0.53801,0.68814,-0.094948,1.3607,-0.20729,0.43439,1.482,-0.38467,-0.19364,0.6572,0.97724,0.30932,0.1266,0.47168,1.3021,0.54091,0.01422,1.5013,0.35522,1.1874,0.022708,0.87857,1.4524,-0.28622,0.19562,0.10891,0.70536,1.1918,-0.26489,0.34809,0.42552,0.25687,1.4226,-0.052176,-0.18748,1.0003,0.68739,1.1387,0.48613,0.18278,1.46,0.095166,0.65267,0.46272,0.045444,0.94684,0.85751,1.0629,0.8721,1.1774,0.67662,1.0326,1.0773,1.2609,-0.26263,0.29083,0.48889,-0.10542,1.3689,1.0632,-0.40383,-0.3075,1.4232,0.76686,0.061157,0.34347,1.1035,-0.24706,1.1067,-0.046772,0.45045,0.042393,0.12437,0.99516,0.10422,0.84668,-0.21414,1.2381,-0.3161,0.68264,1.0499,0.17656,1.5098,-0.022474,-0.34635,-0.26498,0.25231,-0.0041014,0.42799,-0.39604,-0.012161,-0.19077,-0.1616,1.5003,-0.16988,-0.1213,0.37331,0.093202,0.4282,1.174,0.2335,-0.34944,0.49805,0.098059,0.58666,-0.018437,0.093828,0.73488,1.2608,0.62601,1.4921,-0.18038,0.78992,1.1513,1.141,1.2777,1.4175,0.73208,1.0319,0.31195,0.13246,0.67001,0.63496,-0.29206,1.1455,1.0092,1.013,0.10587,-0.017107,0.21474,1.2948,0.86658,0.47383,0.36376,-0.11152,0.91422,1.0224,0.5204,0.9124,1.1179,1.3025,-0.39375,0.45457,0.16312,1.3669,1.1828,1.4073,1.2178,0.85202,0.32887,0.73041,-0.23052,0.33428,1.2272,1.0661,-0.13978,-0.38788,0.33832,-0.40262,-0.33531,0.10016,1.1617,0.88754,-0.094766,-0.34098,-0.00086362,0.98167,1.043,-0.36456,0.10346,1.0014,1.0042,0.62358,1.332,0.012094,0.49288,0.12175,1.0271,0.14778,-0.2036,0.78596,1.3949,1.0443,-0.089024,0.82777,-0.38262,0.31537,0.18875,0.074185,0.76437,1.207,0.31019,1.4883,1.4333,0.47067,-0.3238,1.4267,0.723,1.4339,0.97191,0.66975,1.0837,-0.28681,0.22426,1.0971,-0.38299,0.040189,0.20736,1.3296,1.215,1.0426,1.0512,0.94691,1.0017,0.35744,0.9416,0.63759;

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
//		cout << models[i] << endl;
}
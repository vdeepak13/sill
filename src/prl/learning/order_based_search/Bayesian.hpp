/**
 * \file Bayesian.hpp Teyssier and Koller Order-Based Search Interface
 */

#ifndef PRL_STRUCTURE_SEARCH_ORDER_BASED_BAYESIAN_HPP
#define PRL_STRUCTURE_SEARCH_ORDER_BASED_BAYESIAN_HPP

#include <iostream>

#include <cmath>
#include <prl/learning/order_based_search/Classes.hpp>
#include <prl/learning/order_based_search/Liste.hpp>
#include <prl/learning/order_based_search/constants.hpp>
#include <prl/model/bayesian_graph.hpp>

#include <prl/macros_def.hpp>

char set[26][256] = {"linear.net",
					"alarm3.net",
					"alarm5.net",
					"alarm4.net",
					"alarm2.net",
					"node9.net",
					"node10.net",
					"node11.net",
					"alarm6.net",
					"alarm7.net",
					"alarm8.net",
					"insurance.net",
					"ALARM.NET",
					"hailfinder.net",
					"Barley.net",
					"Diabetes.libb",
					"aviv.csv",
					"subnet4.names",
					"alarm.csv",
					"nursery.csv",
					"letters.csv",
					"edsgc.csv",
					"alarmtot.csv",
					"nurserytot.csv",
					"letterstot.csv",
					"edsgctot.csv"};

char dset[26][256] = {"linear.net",
					"alarm3.net",
					"alarm5.net",
					"alarm4.net",
					"alarm2.net",
					"node9.net",
					"node10.net",
					"node11.net",
					"alarm6.net",
					"alarm7.net",
					"alarm8.net",
					"insurance.net",
					"ALARM.NET",
					"hailfinder.net",
					"Barley.net",
					"Diabetes.libb",
					"aviv.csv",
					"subnet4.data",
					"alarm.csv",
					"nursery.csv",
					"letters.csv",
					"edsgc.csv",
					"alarmtot.csv",
					"nurserytot.csv",
					"letterstot.csv",
					"edsgctot.csv"};

int type[26] = {FULL,FULL,FULL,FULL,FULL,FULL,FULL,FULL,FULL,FULL,FULL,FULL,FULL,FULL,FULL,FULL,CSV,NAME,CSV,CSV,CSV,CSV,CSV,CSV,CSV,CSV};



//What about using real tabu list ???
void GreedyAnalyse()
{
	RealNetwork *bn;
	RealNetwork *bn2;
	Statistics	*stats;
	GreedySearch *search;
	int	maxParent = MAX_PARENT;
	int g,gmax = 16;
	int nb_iterstat = NB_ITER_STAT;
	int nb_iter = NB_ITER;
	int tabu_size = 25;
	int search_depth = SEARCH_DEPTH;
	int nb_restart = NB_RESTART;
	int deviation = DEVIATION;
	int	rmin = 8;
	BOOL * Graph;
	State * maxState;
	int t,iter,iterstat;
	//int seed = clock();
	srand(0);

	double *count = new double[nb_iterstat*nb_iter*gmax];
	double *times = new double[gmax];
	int *nodes = new int[gmax];


#ifdef GRAPH_SPACE
	if (!OBS_SUPPRESS_OUTPUT) printf("GRAPH SPACE\n");
	char scorestring[] = "scoreXX-XX.scg";
#endif

#ifdef ORDER_SPACE
	if (!OBS_SUPPRESS_OUTPUT) printf("ORDER SPACE\n");
	char scorestring[] = "scoreXX-XX.sco";
#endif

	char statstring[] = "statsXX-XX.stt.XXX.XX";

	for (tabu_size = FROM_TABU; tabu_size < TO_TABU ; tabu_size+=TABU_STEP)
	
	{
		for (g = 0; g < gmax; g++)
			times[g] = 0;
	
		for ( iterstat = 0; iterstat < nb_iterstat; iterstat ++)
		{
			for ( iter = 0; iter < nb_iter; iter ++)
			{
				for (g = 0; g < gmax; g++)
				{
					count[iterstat*nb_iter*gmax+iter*gmax+g] = 0;
				}
			}
		}
		
		//for (g = 0; g < gmax; g++)
		{
		
			g = GRAPH_NB	;

			bn = new RealNetwork(set[g], type[g]);

			maxParent = bn->GetMaxParent();
			
			if (!OBS_SUPPRESS_OUTPUT) printf("Nodes = %d   MaxNumParents = %d\n", bn->GetnumVariables(), maxParent);


			for (iterstat = 0; iterstat < nb_iterstat; iterstat ++)
			{
				if (!OBS_SUPPRESS_OUTPUT) printf("=============Iteration Stat %d/%d==============\n",iterstat+1,nb_iterstat);
			


#ifdef GRAPH_SPACE
				sprintf(scorestring,"score%02d-%02d.scg",g,iterstat);
#endif

#ifdef ORDER_SPACE
				sprintf(scorestring,"score%02d-%02d.sco",g,iterstat);
#endif


				sprintf(statstring,"stats%02d-%02d.stt",g,iterstat);

				//stats = new Statistics(statstring);
				//stats = new Statistics(bn, STAT_NB);

				bn2 = new RealNetwork(bn);

				stats = new Statistics(statstring);
				//search = new GreedySearch(scorestring);//bn->GetnumVariables(), maxParent+1, stats);
				search = new GreedySearch(bn->GetnumVariables(), maxParent+1, stats, stats, rmin,UNDEF);
				//Create empty bayesian network from bn (copy labels, num Varibles and domains)
			
					
				for ( iter = 0; iter < nb_iter; iter ++)
				{
					if (!OBS_SUPPRESS_OUTPUT) printf("=============Iteration %d/%d===\n",iter+1,nb_iter);
					if (!OBS_SUPPRESS_OUTPUT) printf("Number of nodes:%d\nMax Parents=%d\n",bn->GetnumVariables(),maxParent);
					if (!OBS_SUPPRESS_OUTPUT) printf("Score : %f\n",bn->GetScore(stats,5));

					t = clock();
			
					nodes[g] = bn->GetnumVariables();
#ifdef GRAPH_SPACE
					maxState = new State(bn->GetnumVariables(), 0, search);
#endif
#ifdef ORDER_SPACE
					maxState = new State(bn->GetnumVariables(), search);
#endif				
					//maxState->Init(search->FamilyTree,search->FamilyCount, search->FamilyList,search->maxParents, search->InFamilyCount , search->InFamilyPtr);

					search->RandomRestartSearch(maxState,nb_restart,deviation,tabu_size,search_depth);

					times[g]+=clock()-t;
					Graph = new BOOL[bn->GetnumVariables()*bn->GetnumVariables()];
					maxState->GetGraph(Graph);
					bn2->SetStructure(Graph);
					count[g + iter*gmax + iterstat*nb_iter*gmax] =bn2->GetScore(stats, 5)-bn->GetScore(stats,5);
					if (!OBS_SUPPRESS_OUTPUT) printf("Score = %f\n", bn2->GetScore(stats, 5));

					bn2->LearnFromStatistics(stats,1);

					bn2->ToFile("result.net");
				
					if (!OBS_SUPPRESS_OUTPUT) printf("Hamming distance : %d\n",bn->GetHDistance(bn2));
#ifdef GRAPH_SPACE
					if (!OBS_SUPPRESS_OUTPUT) printf("Number set values needed : %d\n", search->SetReachedCount );

#endif
					delete maxState;
					delete[] Graph;
				}
				delete search;
				delete bn2;
				delete stats;
			}
			delete  bn;
		}
		//mean = 0.167324 var = 0.729347 min = -0.000000 max = 3.346473
		
		FILE * log;
                if (!OBS_SUPPRESS_OUTPUT) log = fopen("result.log","a");

		double mean,var,min,max,min2,max2;
		double *mean2;

		mean2 = new double[nb_iterstat];

		//for (g = 0; g < gmax; g++)
		{
			times[g] /= (double) nb_iter * CLOCKS_PER_SEC;

			
			mean = 0;
			min =0;
			max = 0;
			for (iterstat = 0; iterstat < nb_iterstat; iterstat ++)
			{
				mean2[iterstat] = 0;
				min2 = SCORE_MAX;
				max2 = SCORE_MIN;
				for (int iter = 0; iter < nb_iter; iter ++)
				{
					mean2[iterstat] +=count[g+ iter*gmax+ iterstat*nb_iter*gmax];
					if (count[g+ iter*gmax+ iterstat*nb_iter*gmax]<min2)
						min2 = count[g+ iter*gmax+ iterstat*nb_iter*gmax];
					if (count[g+ iter*gmax+ iterstat*nb_iter*gmax]>max2)
						max2 = count[g+ iter*gmax+ iterstat*nb_iter*gmax];
				}
				min += min2;
				max += max2;
				mean2[iterstat] /= nb_iter;
				mean += mean2[iterstat] ;
			}
			mean/= nb_iterstat;
			min /= nb_iterstat;
			max /= nb_iterstat;

			var = 0;
			for (iterstat = 0; iterstat < nb_iterstat; iterstat ++)
				for (int iter = 0; iter < nb_iter; iter ++)
					var += (count[g+ iter*gmax + iterstat*nb_iter*gmax]-mean2[iterstat])*(count[g+ iter*gmax+ iterstat*nb_iter*gmax]-mean2[iterstat]);
			var = sqrt(var/(nb_iter*nb_iterstat));
		
	#ifdef ORDER_SPACE
	#ifdef ORDER_SPACE2
			if (!OBS_SUPPRESS_OUTPUT) printf("O2");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"O2");
	#ifdef NO_PLATEAU
			if (!OBS_SUPPRESS_OUTPUT) printf("NP");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"NP");
	#endif
	#ifdef NO_REVERSE
			if (!OBS_SUPPRESS_OUTPUT) printf("NR");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"NR");
	#endif
	#ifdef RANDOM_START
			if (!OBS_SUPPRESS_OUTPUT) printf("RS");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"RS");
	#endif
			if (!OBS_SUPPRESS_OUTPUT) printf(" ");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log," ");
	#else

	#ifdef ORDER_SPACE3
			if (!OBS_SUPPRESS_OUTPUT) printf("O3");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"O3");
	#ifdef RANDOM_START
			if (!OBS_SUPPRESS_OUTPUT) printf("RS");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"RS");
	#endif
			if (!OBS_SUPPRESS_OUTPUT) printf(" ");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log," ");
	#else
			if (!OBS_SUPPRESS_OUTPUT) printf("O");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"O");
	#ifdef RANDOM_START
			if (!OBS_SUPPRESS_OUTPUT) printf("RS");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"RS");
	#endif
			if (!OBS_SUPPRESS_OUTPUT) printf(" ");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log," ");
	#endif
	#endif
	#else
			if (!OBS_SUPPRESS_OUTPUT) printf("G");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"G");
	#ifdef RANDOM_START
			if (!OBS_SUPPRESS_OUTPUT) printf("RS");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"RS");
	#endif
			if (!OBS_SUPPRESS_OUTPUT) printf(" ");
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"	");
	#endif
			if (!OBS_SUPPRESS_OUTPUT) printf("nodes= %d maxparent=%d stat= %d iter= %d restart= %d deviation= %d tabu= %d depth= %d mean= %f var= %f min= %f max= %f t= %f\n",
				nodes[g], maxParent, nb_iterstat, nb_iter, nb_restart,deviation, tabu_size, search_depth,mean,var,min,max,times[g]);
			if (!OBS_SUPPRESS_OUTPUT) fprintf(log,"nodes=	%d	maxparent=	%d	stat=	%d	iter=	%d	restart=	%d	deviation=	%d	tabu=	%d	depth=	%d	mean=	%f	var=	%f	min=	%f	max=	%f	t=	%f\n",
				nodes[g], maxParent, nb_iterstat, nb_iter, nb_restart,deviation, tabu_size, search_depth,mean,var,min,max,times[g]);
		}

		if (!OBS_SUPPRESS_OUTPUT) fclose(log);
		delete mean2;
	}
	
	delete count;
	delete times;
	delete nodes;
	//mean = 508.881698 var = 32.988468 min = 450.941438 max = 585.059754
	//mean = 117.581852 var = 287.967933 min = -521.674439 max = 480.416117
	//mean = 177.745165 var = 293.268111 min = -440.109267 max = 521.674294
}

void MakeStats(int g, int nb, int stat_nb)
{

	char statstring[] = "statsXX-XX-XXXXXXXX.stt.xxxx";

	RealNetwork *bn;		

	Statistics *stats;


	bn = new RealNetwork(set[g], type[g]);

	for (int iterstat = 0; iterstat < nb; iterstat ++)
	{

			sprintf(statstring,"stats%02d-%02d-%08d.stt",g,iterstat, stat_nb);

			stats = new Statistics(bn,stat_nb);

			stats->toFile(statstring);
			
			delete stats;
			
			sprintf(statstring,"stats%02d-%02d-%08d.stt.tst",g,iterstat, stat_nb);

			stats = new Statistics(bn,stat_nb);

			stats->toFile(statstring);
			
			delete stats;
	}

	delete bn;

}

/*void MakeScores(int iterstat)
{
	char statstring[] = "statsXX-XX-XXXXXXXX.stt";

	RealNetwork *bn;		

	Statistics *stats;
	GreedySearch *search;

	int g = GRAPH_NB;

	bn = new RealNetwork(set[g]);

	//for (int iterstat = 0; iterstat < nb; iterstat ++)
	{

			sprintf(statstring,"stats%02d-%02d.stt",g,iterstat);

#ifdef GRAPH_SPACE
			sprintf(scorestring,"score%02d-%02d.scg",g,iterstat);
#endif

#ifdef ORDER_SPACE
			sprintf(scorestring,"score%02d-%02d.sco",g,iterstat);
#endif

			stats = new Statistics( statstring );

			search = new GreedySearch( stats->GetnumVariables(), bn->GetMaxParent()+1, stats);

			//search->ToFile( scorestring );
			
			delete search;
			delete stats;
	}

	delete bn;


}*/


void GreedyTimeAnalyse(int g, int numsamp, int timelimit, int nb_iter, int nb_iterstat, int search_depth, int deviation, int tabu_size, int rmin, int numCandidates)
{
	RealNetwork *bn;
	RealNetwork *bn2;
	Statistics	*stats1;
	Statistics	*stats2;
	GreedySearch *search;
	
	BOOL * Graph;
	State * maxState;
	int iter,iterstat;
	char	graphName[32];
	//int seed = clock();
	srand(clock());

	double *count = new double[nb_iterstat*nb_iter];
	int	   *countHamming = new int[nb_iterstat*nb_iter];
	int	   *countEdges = new int[nb_iterstat*nb_iter];

//#ifdef GRAPH_SPACE
	int	   *countSetValues = new int[nb_iterstat*nb_iter];

//#endif
	

	char statstring[] = "statsXX-XX-XXXXXXXX.stt.XXX.XX";

	for ( iterstat = 0; iterstat < nb_iterstat; iterstat ++)
	{
		for ( iter = 0; iter < nb_iter; iter ++)
		{
			count[iterstat*nb_iter+iter] = 0;
			countHamming[iterstat*nb_iter+iter] = 0;
			countEdges[iterstat*nb_iter+iter] = 0;
		}
	}
		
	//int numsamp = UNDEF;

	strcpy(graphName, set[g]);
	strtok(graphName,".");

	bn = new RealNetwork(set[g], type[g]); //load the real net from file

	int	maxParent = bn->GetMaxParent(); //get max in-degree of the real net
	
	if (!OBS_SUPPRESS_OUTPUT) printf("%s	",graphName);
#ifdef GRAPH_SPACE
	if (!OBS_SUPPRESS_OUTPUT) printf("DAG\n");
#else
	if (!OBS_SUPPRESS_OUTPUT) printf("ORD\n");
#endif
	if (!OBS_SUPPRESS_OUTPUT) printf("Nodes=	%d	MaxNumParents=	%d	NumberEdges=	%d	NbSamples=	%d	", bn->GetnumVariables(), maxParent, bn->GetNumEdges(), numsamp);

	double baseScore = 0;
	for (iterstat = 0; iterstat < nb_iterstat; iterstat ++) //for all training sets..
	{
		sprintf(statstring,"stats%02d-%02d-%08d.stt",g,iterstat, numsamp);
		stats1 = new Statistics(statstring);
		baseScore += bn->GetScore(stats1, VIRTUAL_COUNT);
		if (!OBS_SUPPRESS_OUTPUT) printf("AA=	%f	",stats1->GetAArity());
		delete stats1;
	}
	
	if (!OBS_SUPPRESS_OUTPUT) printf("BaseScore=	%f\n", baseScore/nb_iterstat);
	
	if (!OBS_SUPPRESS_OUTPUT) printf("Tabu=	%d	Depth=	%d	Deviation=	%d	TimeLimit=	%d	nbStat=	%d	nbIter=	%d	Rmin=	%d	Candidates=	%d\n", tabu_size, search_depth, deviation, timelimit, nb_iterstat, nb_iter, rmin, numCandidates);	

	for (iterstat = 0; iterstat < nb_iterstat; iterstat ++) //for all training sets..
	{
		//if (!OBS_SUPPRESS_OUTPUT) printf("=============Iteration Stat %d/%d==============\n",iterstat+1,nb_iterstat);
	
		sprintf(statstring,"stats%02d-%02d-%08d.stt",g,iterstat, numsamp);
		stats1 = new Statistics(statstring);
		sprintf(statstring,"stats%02d-%02d-%08d.stt.tst",g,iterstat, numsamp);
		stats2 = new Statistics(statstring);
	
		bn2 = new RealNetwork(bn);

	
		assert( numsamp == stats1->GetnumSamples() );

		//Create empty bayesian network from bn (copy labels, num Varibles and domains)
		for ( iter = 0; iter < nb_iter; iter ++) //for several random retries
		{
			if (!OBS_SUPPRESS_OUTPUT) printf("=============Iteration %d/%d %d/%d===\n",iter+1,nb_iter, iterstat+1,nb_iterstat);
		
			search = new GreedySearch(bn->GetnumVariables(), maxParent+1, stats1, stats2, rmin, numCandidates);
#ifdef GRAPH_SPACE
			maxState = new State(bn->GetnumVariables(), 0, search);
#endif

#ifdef ORDER_SPACE
			maxState = new State(bn->GetnumVariables(), search);
#endif
			search->RandomRestartSearch(maxState,timelimit,deviation,tabu_size,search_depth);

			Graph = new BOOL[bn->GetnumVariables()*bn->GetnumVariables()];
			maxState->GetGraph(Graph);
			bn2->SetStructure(Graph);
			count[iter + iterstat*nb_iter] =bn2->GetScore(stats1, VIRTUAL_COUNT);
			countHamming[iter + iterstat*nb_iter] = bn->GetHDistance(bn2);
			countEdges[iter + iterstat*nb_iter] = bn2->GetNumEdges();
//#ifdef GRAPH_SPACE
			countSetValues[iter + iterstat*nb_iter] = search->SetReachedCount;
//#endif
			bn2->LearnFromStatistics(stats1,1);
			bn2->ToFile("result.net");
	
			delete maxState;
			delete[] Graph;
			delete search;
		}
	delete bn2;
	delete stats1;
	delete stats2;
	}
	delete  bn;


	double mean,var,min,max,min2,max2,meanh, meane, means;
	double *mean2;

	mean2 = new double[nb_iterstat];

			
			
	mean = 0;
	meanh = 0;
	meane = 0;
	means = 0;
	min =0;
	max = 0;
	for (iterstat = 0; iterstat < nb_iterstat; iterstat ++)
	{
		mean2[iterstat] = 0;
		min2 = SCORE_MAX;
		max2 = SCORE_MIN;
		for (int iter = 0; iter < nb_iter; iter ++)
		{
			meanh += countHamming[iter+ iterstat*nb_iter];
			meane += countEdges[iter+ iterstat*nb_iter];
	
//#ifdef GRAPH_SPACE
			means += countSetValues[iter+ iterstat*nb_iter];
//#endif
			mean2[iterstat] +=count[iter+ iterstat*nb_iter];
			if (count[iter+ iterstat*nb_iter]<min2)
				min2 = count[iter+ iterstat*nb_iter];
			if (count[iter+ iterstat*nb_iter]>max2)
				max2 = count[iter+ iterstat*nb_iter];
		}
		min += min2;
		max += max2;
		mean2[iterstat] /= nb_iter;
		mean += mean2[iterstat] ;
	}
	mean/= nb_iterstat;
	min /= nb_iterstat;
	max /= nb_iterstat;
	meanh /= nb_iterstat*nb_iter;
	meane /= nb_iterstat*nb_iter;
//#ifdef GRAPH_SPACE
	means /= nb_iterstat*nb_iter;
//#endif
	var = 0;
	for (iterstat = 0; iterstat < nb_iterstat; iterstat ++)
		for (int iter = 0; iter < nb_iter; iter ++)
			var += (count[ iter + iterstat*nb_iter]-mean2[iterstat])*(count[iter+ iterstat*nb_iter]-mean2[iterstat]);
	var = sqrt(var/(nb_iter*nb_iterstat));

	if (!OBS_SUPPRESS_OUTPUT) printf("mean=	%f	max=	%f	min=	%f	var=	%f	Hamming=	%f	NumEdges=	%f	",mean,max,min,var,meanh,meane);

// GRAPH_SPACE
	if (!OBS_SUPPRESS_OUTPUT) printf("setcomped=	%f", means);

//#endif
	if (!OBS_SUPPRESS_OUTPUT) printf("\n");

	delete count;
	delete countHamming;
	delete countEdges;
	delete countSetValues;
	delete mean2;

}


void GreedyDataTimeAnalyse(int g, int timelimit, int nb_iter, int nb_iterstat, int search_depth, int deviation, int tabu_size, int rmin, int numCandidates, int	maxParent)
{
	
	RealNetwork *bn;
	RealNetwork *bn2;
	Statistics	*stats1;
	Statistics	*stats2;
	GreedySearch *search;
	
	BOOL * Graph;
	State * maxState;
	int iter,iterstat;
	char	graphName[32];
	char statstring[] = "statsXX-XX-XXXXXXXX.stt.XXX.XX";
	//int seed = clock();
	srand(0);

	double *count = new double[nb_iterstat*nb_iter];
	//int	   *countHamming = new int[nb_iterstat*nb_iter];
	int	   *countEdges = new int[nb_iterstat*nb_iter];

//#ifdef GRAPH_SPACE
	int	   *countSetValues = new int[nb_iterstat*nb_iter];

//#endif
	


	for ( iterstat = 0; iterstat < nb_iterstat; iterstat ++)
	{
		for ( iter = 0; iter < nb_iter; iter ++)
		{
			count[iterstat*nb_iter+iter] = 0;
	//		countHamming[iterstat*nb_iter+iter] = 0;
			countEdges[iterstat*nb_iter+iter] = 0;
		}
	}
		
	//int numsamp = UNDEF;

	strcpy(graphName, set[g]);
	strtok(graphName,".");

	bn = new RealNetwork(set[g], type[g]);

	
	if (!OBS_SUPPRESS_OUTPUT) printf("%s	",graphName);
#ifdef GRAPH_SPACE
	if (!OBS_SUPPRESS_OUTPUT) printf("DAG\n");
#else
	if (!OBS_SUPPRESS_OUTPUT) printf("ORD\n");
#endif
	if (!OBS_SUPPRESS_OUTPUT) printf("Nodes=	%d	MaxNumParents=	%d	NumberEdges=	%d	", bn->GetnumVariables(), maxParent, bn->GetNumEdges());

	/*double baseScore = 0;
	for (iterstat = 0; iterstat < nb_iterstat; iterstat ++)
	{
		sprintf(statstring,"stats%02d-%02d-%08d.stt",g,iterstat, numsamp);
		stats = new Statistics(statstring);
		baseScore += bn->GetScore(stats, VIRTUAL_COUNT);
		delete stats;
	}*/
	
	//if (!OBS_SUPPRESS_OUTPUT) printf("BaseScore=	%f\n", baseScore/nb_iterstat);
	
	if (!OBS_SUPPRESS_OUTPUT) printf("Tabu=	%d	Depth=	%d	Deviation=	%d	TimeLimit=	%d	nbStat=	%d	nbIter=	%d	Rmin=	%d	Candidates=	%d\n", tabu_size, search_depth, deviation, timelimit, nb_iterstat, nb_iter, rmin, numCandidates);	

	for (iterstat = 0; iterstat < nb_iterstat; iterstat ++)
	{
		//if (!OBS_SUPPRESS_OUTPUT) printf("=============Iteration Stat %d/%d==============\n",iterstat+1,nb_iterstat);
	
		sprintf(statstring,"%s.trn.%d",dset[g],iterstat+1);
		stats1 = new Statistics(statstring, bn, type[g]);
		sprintf(statstring,"%s.tst.%d",dset[g],iterstat+1);
		stats2 = new Statistics(statstring, bn, type[g]);
	
		if (!OBS_SUPPRESS_OUTPUT) printf("AA=	%f	\n",stats1->GetAArity());
		bn2 = new RealNetwork(bn);

	
		//assert( numsamp == stats->GetnumSamples() );

		//Create empty bayesian network from bn (copy labels, num Varibles and domains)
		for ( iter = 0; iter < nb_iter; iter ++)
		{
			if (!OBS_SUPPRESS_OUTPUT) printf("=============Iteration %d/%d %d/%d===\n",iter+1,nb_iter, iterstat+1,nb_iterstat);
		
			
			search = new GreedySearch(bn->GetnumVariables(), maxParent+1, stats1, stats2, rmin, numCandidates);
#ifdef GRAPH_SPACE
			maxState = new State(bn->GetnumVariables(), 0, search);
#endif

#ifdef ORDER_SPACE
			maxState = new State(bn->GetnumVariables(), search);
#endif
			search->RandomRestartSearch(maxState,timelimit,deviation,tabu_size,search_depth);

			Graph = new BOOL[bn->GetnumVariables()*bn->GetnumVariables()];
			maxState->GetGraph(Graph);
			bn2->SetStructure(Graph);
			count[iter + iterstat*nb_iter] =bn2->GetScore(stats1, VIRTUAL_COUNT);
			//countHamming[iter + iterstat*nb_iter] = bn->GetHDistance(bn2);
			countEdges[iter + iterstat*nb_iter] = bn2->GetNumEdges();
//#ifdef GRAPH_SPACE
			countSetValues[iter + iterstat*nb_iter] = search->SetReachedCount;
//#endif
			bn2->LearnFromStatistics(stats1,1);
			bn2->ToFile("result.net");
	
			delete maxState;
			delete[] Graph;
			delete search;
		}
	delete bn2;
	delete stats1;
	delete stats2;
	}
	delete  bn;


	double mean,var,min,max,min2,max2,meanh, meane, means;
	double *mean2;

	mean2 = new double[nb_iterstat];

			
			
	mean = 0;
	meanh = 0;
	meane = 0;
	means = 0;
	min =0;
	max = 0;
	for (iterstat = 0; iterstat < nb_iterstat; iterstat ++)
	{
		mean2[iterstat] = 0;
		min2 = SCORE_MAX;
		max2 = SCORE_MIN;
		for (int iter = 0; iter < nb_iter; iter ++)
		{
			//meanh += countHamming[iter+ iterstat*nb_iter];
			meane += countEdges[iter+ iterstat*nb_iter];
	
//#ifdef GRAPH_SPACE
			means += countSetValues[iter+ iterstat*nb_iter];
//#endif
			mean2[iterstat] +=count[iter+ iterstat*nb_iter];
			if (count[iter+ iterstat*nb_iter]<min2)
				min2 = count[iter+ iterstat*nb_iter];
			if (count[iter+ iterstat*nb_iter]>max2)
				max2 = count[iter+ iterstat*nb_iter];
		}
		min += min2;
		max += max2;
		mean2[iterstat] /= nb_iter;
		mean += mean2[iterstat] ;
	}
	mean/= nb_iterstat;
	min /= nb_iterstat;
	max /= nb_iterstat;
	//meanh /= nb_iterstat*nb_iter;
	meane /= nb_iterstat*nb_iter;
//#ifdef GRAPH_SPACE
	means /= nb_iterstat*nb_iter;
//#endif
	var = 0;
	for (iterstat = 0; iterstat < nb_iterstat; iterstat ++)
		for (int iter = 0; iter < nb_iter; iter ++)
			var += (count[ iter + iterstat*nb_iter]-mean2[iterstat])*(count[iter+ iterstat*nb_iter]-mean2[iterstat]);
	var = sqrt(var/(nb_iter*nb_iterstat));

	if (!OBS_SUPPRESS_OUTPUT) printf("mean=	%f	max=	%f	min=	%f	var=	%f	NumEdges=	%f	",mean,max,min,var,meane);

// GRAPH_SPACE
	if (!OBS_SUPPRESS_OUTPUT) printf("setcomped=	%f", means);

//#endif
	if (!OBS_SUPPRESS_OUTPUT) printf("\n");

	delete count;
	//delete countHamming;
	delete countEdges;
	delete countSetValues;
	delete mean2;

}

/**
 * Use Order Based Search to find the structure of a Bayes net which describes
 * the data in the statistics files.
 *
 * @todo Figure out exactly how the training and tests sets are used.
 * @todo Change this so that it returns the Bayes net model, not just the
 *       model's structure.
 */
prl::bayesian_graph<prl::finite_variable*>
GreedyAnalyse(const prl::assignment_dataset& trainStatsDS,
              const prl::assignment_dataset& testStatsDS,
              int timelimit, int search_depth, int deviation,
              int tabu_size, int rmin, int numCandidates,
              int maxParent) {
  using namespace prl;
  Statistics	*stats1;
  Statistics	*stats2;
  GreedySearch *search;

  State * maxState;
  srand(0);

  stats1 = new Statistics(trainStatsDS);
  stats2 = new Statistics(testStatsDS);

  assert( stats1->GetnumVariables() == stats2->GetnumVariables() );

  search = new GreedySearch( stats1->GetnumVariables(), maxParent+1, stats1, stats2, rmin, numCandidates);
  maxState = new State(stats1->GetnumVariables(), search);
  search->RandomRestartSearch(maxState,timelimit,deviation,tabu_size,search_depth);

  BOOL * Graph;
  Graph = new BOOL[stats1->GetnumVariables()*stats1->GetnumVariables()];
  maxState->GetGraph(Graph);
  // Graph is now a vectorized matrix where
  //  (i,j)=TRUE iff node i is a parent of node j

  // Now, create a PRL Bayes net graph.
  finite_var_vector varvec;
  foreach(finite_variable* v, trainStatsDS.finite_variables())
    varvec.push_back(v);
  bayesian_graph<finite_variable*> bg(trainStatsDS.finite_variables());
  for (int i = 0; i < stats1->GetnumVariables(); i++) {
    for (int j = 0; j < stats1->GetnumVariables(); j++) {
      if (Graph[i * stats1->GetnumVariables() + j])
        bg.add_edge(varvec[i],varvec[j]);
    }
  }

  return bg;
}

#include <prl/macros_undef.hpp>

#endif // PRL_STRUCTURE_SEARCH_ORDER_BASED_BAYESIAN_HPP

// Classes.h: interface for the CClasses class.
//
//////////////////////////////////////////////////////////////////////


#ifndef PRL_ORDER_BASED_SEARCH_CLASSES_HPP
#define PRL_ORDER_BASED_SEARCH_CLASSES_HPP

#include <prl/datastructure/dense_table.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/learning/dataset/assignment_dataset.hpp>
#include <prl/learning/order_based_search/constants.hpp>
#include <prl/variable.hpp>

#include <string>
#include <cmath>
#include <iostream>

#include <prl/macros_def.hpp>

#define OBS_SUPPRESS_OUTPUT 1 // Set = 0 to print stuff out while this runs.

/** Gamma function in double precision **/
/*
static double dgamma(double x)
{
    int k, n;
    double w, y;

    n = x < 1.5 ? -((int) (2.5 - x)) : (int) (x - 1.5);
    w = x - (n + 2);
    y = ((((((((((((-1.99542863674e-7 * w + 1.337767384067e-6) * w - 
        2.591225267689e-6) * w - 1.7545539395205e-5) * w + 
        1.45596568617526e-4) * w - 3.60837876648255e-4) * w - 
        8.04329819255744e-4) * w + 0.008023273027855346) * w - 
        0.017645244547851414) * w - 0.024552490005641278) * w + 
        0.19109110138763841) * w - 0.233093736421782878) * w - 
        0.422784335098466784) * w + 0.99999999999999999;
    if (n > 0) {
        w = x - 1;
        for (k = 2; k <= n; k++) {
            w *= x - k;
        }
    } else {
        w = 1;
        for (k = 0; k > n; k--) {
            y *= x - k;
        }
    }
    return w / y;
};
*/

//Compute C(n,p)
static	int	ComputeCombin(int n, int p)
{
	int k;
	int	res = 1;
	for (k = 0; k < p ; k++)
		res *= (n-k);
	for ( k = 2; k < p + 1 ; k++)
		res /= k;
	return res;
}

static int	Combin[MAXNODES][MAXNODES];

static void	PopulateCombin()
{
	int i,j;

	for ( i = 0; i< MAXNODES; i++)
		for ( j = 0; j < MAXNODES ; j++)
			if ( j>i )
				Combin[i][j] = 0;
			else
				Combin[i][j] = ComputeCombin(i,j);
}

//compute the next assignement by adding one from the left to the right
//the int returned in the var nb that has been changed
//if it is equal to numvars then we have reached th end of the assignment
//adding from left to right
static	int	NextAssignment(int numvars, int * assignment, int * domains)
{
	int	j = 0;

	while ( j < numvars )
	{
		assignment[j]++;

		if (assignment[j] == domains [j])
		{
			j++;
			for ( int jj = 0; jj < j; jj++)
				assignment[jj] = 0;
		}
		else
			break;
	}

	return j;

}



//compute the next assignement by adding one from the left to the right
//the int returned in the var nb that has been changed
//if it is equal to numvars then we have reached th end of the assignment
//adding from right to left
static	int	NextAssignmentRight(int numvars, int * assignment, int * domains)
{
	int	j = numvars - 1;

	while ( j >= 0 )
	{
		assignment[j]++;

		if (assignment[j] == domains [j])
		{
			j--;
			for ( int jj = numvars - 1; jj > j; jj--)
				assignment[jj] = 0;
		}
		else
			break;
	}

	return j;

}

//Function mapping:   O(k)
//(x0<x1<...<xk-1) -> [0..C(n,k)-1] with xi in [0..n-1]
static int	SetToPtr(int n, int k, int *x)
{
	int j;
	int ptr = 0;
	for ( j = 0; j < k; j++ )
	{
		ptr += Combin[ n-1-x[j] ][ k-j ];
	}
	return  Combin[n][k] - ptr - 1;

}

//Function mapping:		O(n)
//(x0<x1<...<xk-1) <- [0..C(n,k)-1] with xi in [0..n-1]
static void	PtrToSet(int ptr, int n, int k, int *x)
{
	int ptr2 = Combin[n][k] - ptr - 1;
	int	v = 0;
	int j;
	for ( j = 0; j < k; j++ )
	{
		while ( ptr2 < Combin[ n-v-1 ][ k-j ])
			v++;
		
		x[j] = v;

		ptr2 -= Combin[ n-v-1 ][ k-j ];

		v++;//x[j-1] can't be equal to x[j] !


	}
}

static int	NextSet(int n, int k, int* x)
{
	int jj;
	int	j = k - 1;

	for ( j = k-1 ; j>=0 ; j--)
	{
		x[j]++;
		if (x[j] != n + j - k + 1)
		{
			for ( jj = j + 1; jj < k; jj++)
				x[jj] = x[jj-1] + 1;
			break;
		}
	}

	return j;

}

/*
static void GetConnexity(BOOL *Graph, BOOL *Connexity, int numVariables)
{
	int ii,jj,x;
	for ( ii = 0; ii < numVariables; ii++)
	{
		for ( jj = 0; jj < numVariables; jj++)
		{
			if (ii != jj)
				Connexity[ii*numVariables+jj]=Graph[ii*numVariables+jj];
			else
				Connexity[ii*numVariables+jj]=TRUE;
		}
	}

	for ( x = 0; x < numVariables; x++)
	{
		for ( ii = 0; ii < numVariables; ii++)
		{
			if (Connexity[ii*numVariables+x])
			{
				for ( jj = 0; jj < numVariables; jj++)
				{
					if (Connexity[x*numVariables+jj])
					{
						assert( ii==jj || !Connexity[jj*numVariables+ii]);
						Connexity[ii*numVariables+jj]=TRUE;
					}
				}
			}
		}
	}

}
*/

/*static void GetAddEdgeConnexity(BOOL *Graph, BOOL *Connexity, int numVariables, int from, int to)
{
	int ii,jj;
	for (jj = 0; jj < numVariables; jj++)
	{
		if (Connexity[jj*numVariables+from])
		{
			for ( ii = 0; ii < numVariables; ii++)
			{
				if (Connexity[to*numVariables+ii])
				{
			
						Connexity[jj*numVariables+ii]=TRUE;
				}
			}
		}
	}

}*/

class Var
{
	//int		numVar;//number corresponding to the variable in the network
	int		domain;//domain size of the variable

public:
	Var(){ domain = 0;}
	void SetDomain(int d){ domain = d; };
	int GetDomain(){ return domain; };
	void CopyFrom(Var * var)
	{
		domain = var->GetDomain();
	}
};

class Network;
class Statistics;
class Counter;

class Factor 
{
protected:	

	int		numVariables;
	int*	Variables;
	int*	Domains;

	int		tablesize;
	double*	table;//table[i]=table[sum ai*ei] with ei=product j<i |Dj|

public:
	Factor()
	{
		numVariables = 0;
		Variables = NULL;
		Domains = NULL;

		//initialize factor to factor 1
		tablesize = 1;
		table = new double[tablesize];
	}

	Factor(Factor * f)
	{
		CopyFrom(f);
	}

	~Factor()
	{
		delete	Variables;
		delete	Domains;
		delete	table;
	}


	//void	SetParentNetwork(Network * net){ parentNet = net; };

	int		GetnumVariables(){ return numVariables;};
	void	SetnumVariables(int n){ numVariables = n;};

	void	SetScope(int* vars, Network * net);

        //! Convert assignment to index in table factor.
        int GetPos(int* assignment)
	{

		if (numVariables == 0)
			return 0;

		int p=assignment[numVariables-1];

		for (int i = numVariables - 2 ; i >= 0 ; i--)
		{

			p *= Domains[i];
			p += assignment[i];
				
		}

		//DEBUG
		//if ( p >= tablesize)
		//	p++;

		ASSERT( p < tablesize );
		return p;

	}

	int		GetSize(){
		return tablesize;
	};
	void	SetValue(double v, int* assignment)	{ table[GetPos(assignment)] = v; };
	double	GetValue(int* assignment){ return table[GetPos(assignment)]; };
	void	GetScope(int* scope){ memcpy(scope,Variables,sizeof(int)*numVariables);};
	void	GetDomains(int* domains){ memcpy(domains,Domains,sizeof(int)*numVariables);};
	void	GetCPT(double * t){ memcpy(t,table,sizeof(double)*tablesize);};
	int		GetVariable(int v){return Variables[v];};
	int		GetDomain(int v){return Domains[v];};
	int*	GetScope(){return Variables;};
	int*	GetDomains(){return Domains;};

//	BOOL	isValidCPT();//Assume first variable is the not given variable
BOOL	isValidCPT()
{
	int*	a=new int[numVariables];
	BOOL	r = TRUE;
	double	p;

	int j;
	int i;

	for ( j = 1; j < numVariables; j++)
		a[j] = 0;


	do
	{

		p = 0;
		
		for ( i = 0 ; i < Domains[0]; i++)
		{
			a[0] = i;
			p += GetValue(a);
		}

		if ( p < 0.999 || p > 1.001)
			r = FALSE;

		j = NextAssignment(numVariables, a, Domains);

	} while (r && j < numVariables);

	delete a;

	return r;
}

	void	CopyFrom(Factor* factor)
	{
		numVariables = factor->GetnumVariables();

		delete	Variables;
		delete	Domains;
		delete	table;

		tablesize = factor->GetSize();
		Variables = new int[numVariables];
		Domains = new int[numVariables];
		table = new double[tablesize];
		
		factor->GetScope(Variables);
		factor->GetDomains(Domains);
		factor->GetCPT(table);

	}

        //! Print differences between this factor and another.
	void	printComparison(Factor* factor);

	void	multiply(Factor* factor);
	void	divide(Factor* factor);
	void	sumOut(int numv2, int* scope2 );

	void	PrintScope()
	{
		for (int i = 0; i < numVariables ; i++)
			if (!OBS_SUPPRESS_OUTPUT) printf("%d ",Variables[i]);
		if (!OBS_SUPPRESS_OUTPUT) printf("\n");
	}
	
	void AddValue(double v, int * assignment){	table[GetPos(assignment)] += v;};
};

//! 1-variable node in Bayes net.
class Node : public Factor
{
protected:
	Var		Variable;//Variable corresponding to this node

public:
	Node(){};
	~Node(){};

	Var*	GetNodeVariable(){ return &Variable; };
	void	SetVariableDomain(int d){ Variable.SetDomain(d); };
	int		GetVariableDomain(){ return Variable.GetDomain(); };
	
	void	Sample(int * values)
	{
		int d = Domains[0];
		int	*a = new int[numVariables];
		double p;
		int v,k;

		for ( v = 1; v < numVariables ; v++)
			a[v] = values[Variables[v]];

		p = RANDOM;

		for ( k = 0; k < d ; k++)
		{
			a[0] = k;
			p -= GetValue(a);
			if (p <= 0)
				break;

		}

		if (k == d)
			values[Variables[0]] = d-1;
		else
			values[Variables[0]] = k;

		delete	a;
		ASSERT(p<0.001);
		
	}

	void LearnCPT(Statistics * stats, double initValue);

	void CopyFrom(Node * node)
	{
		Variable.CopyFrom(&node->Variable);
		Factor::CopyFrom((Factor*)node);
	}

	void printComparison(Node * node)
	{
		if (GetVariableDomain() != node->GetVariableDomain())
		{
			if (!OBS_SUPPRESS_OUTPUT) printf("Variable disagree on domain size.\n");
			return;
		}

		Factor::printComparison((Factor*)node);
		
	}

	double	GetScore(Statistics *stats, double initValue);
	//double	GetScore(GreedySearch * search);
};



class Network 
{

protected:

	int		numVariables;
	Node*	Nodes;

public:


	Network()
	{
		numVariables = 0;
		Nodes=NULL;
	}

	~Network()
	{
		delete[]	Nodes;
	}
	
	int GetnumVariables(){ return numVariables;};
	Node*	GetNode(int n){ return &Nodes[n];};
	
	void	GetDomains(int * d)
	{
		for (int i = 0; i < numVariables ; i++)
			d[i] = Nodes[i].GetVariableDomain();
	}

        void GetTopologicalOrdering(int * ord)
        {
          BOOL	*assigned=new BOOL[numVariables];
          int		ssize,v,v2,vv,c;
          
          for (v = 0 ; v < numVariables; v++)
            assigned[v] = FALSE;
          
          for ( c = 0; c < numVariables; c++)
            {
              for (v = 0 ; v < numVariables; v++)
		{
                  if (!assigned[v])
                    {
                      ssize = Nodes[v].GetnumVariables();
                      
                      for ( v2 = 0; v2 < ssize ; v2++)
                        {
                          vv = Nodes[v].GetVariable(v2);
                          if (vv != v && !assigned[vv])
                            break;
                        }
                      
                      if ( v2 == ssize)//we have found a fully asigned variable: v
                        break;
                    }
		}
              
              ASSERT( v!= numVariables);
              
              assigned[v] = TRUE;
              ord[c] = v;
            }
          
          delete assigned;
        }

	void	ForwardSample(int * sample, int * ord)//ord:topological ordering
	{
		for (int v = 0; v < numVariables ; v++)
			Nodes[ord[v]].Sample(sample);
	}

	
	void	MakeStatistics(char * filename, int nb)//ord:topological ordering
	{
		int	*ord = new int[numVariables];
		int *samp = new int[nb*numVariables];
		int *dom = new int[numVariables];

		FILE * f=fopen(filename,"w");

		GetTopologicalOrdering(ord);
		GetDomains(dom);


		for (int n = 0; n < nb ; n++ )
			ForwardSample( &samp[n*numVariables], ord );
		

		fwrite(&numVariables,sizeof(int),1,f);
		
		fwrite(&dom,sizeof(int) * numVariables,1,f);

		fwrite(&nb,sizeof(int),1,f);

		fwrite(samp, sizeof(int), numVariables*nb, f);

		fclose(f);

		delete ord;
		delete samp;
		delete dom;

	}

        void LearnFromStatistics(Statistics * stats, double initValue)
        {
          for (int i = 0; i < numVariables ; i++ )
            Nodes[i].LearnCPT(stats, initValue);
        }

	Network(Network * net)//copy the structure but not the values of CPTs
	{
		numVariables = net->numVariables;

		Nodes = new Node[numVariables];
		for (int n = 0; n < numVariables; n++)
			Nodes[n].CopyFrom(net->GetNode(n));
	}

	void printComparison(Network * net)
	{
		
		if (numVariables != net->GetnumVariables())
		{
			if (!OBS_SUPPRESS_OUTPUT) printf("Not same number of variables\n");
			return;
		}

		for (int n = 0; n < numVariables; n++)
		{
			if (!OBS_SUPPRESS_OUTPUT) printf("Node %d:\n",n);
			Nodes[n].printComparison(net->GetNode(n));
		}

	}

	//only the nodes varaibles are built (with their domain)
	//and we are given the structure of the network : graph[i*numVariable+j] == TRUE <=> i->j
	//we build the Factors
	void SetStructure(BOOL *graph)
	{
		int c,i,n;
		int *scope;

		for ( n = 0; n < numVariables;n++ )
		{
			//count the number of parents
			c = 1;
			for ( i = 0; i < numVariables; i++ )
			{
				if (graph[i*numVariables+n])
					c++;
			}

			Nodes[n].SetnumVariables(c);

			scope = new int[c];
			scope[0] = n;

			c = 1;
			for (i = 0; i < numVariables; i++ )
			{
				if (graph[i*numVariables+n])
					scope[c++] = i;
			}

			Nodes[n].SetScope(scope, this);

			delete scope;

		}

	}

	void GetStructure(BOOL *graph)
	{
		int *scope;

		for (int n = 0; n < numVariables;n++ )
		{
			for (int n2 = 0; n2 < numVariables;n2++ )
				graph[n2*numVariables+n] = FALSE;
			//count the number of parents
			scope = Nodes[n].GetScope();
			for (int i = 0; i < Nodes[n].GetnumVariables(); i++ )
				graph[scope[i]*numVariables+n] = TRUE;
		}

	}
	double	GetScore(Statistics *stats, double initValue)
	{
		double	score = 0;
		for (int i = 0; i < numVariables ; i++ )
		{
			//DEBUG
			//double score2 = Nodes[i].GetScore(stats,1);
			//printf("Family(%d) : %f\n", i, score2);
			score += Nodes[i].GetScore(stats,initValue);
		}
		return score;
	}

/*	double	GetScore(GreedySearch *search, double initValue)
	{
		double	score = 0;
		for (int i = 0; i < numVariables ; i++ )
		{
			score += Nodes[i].GetScore(search);
		}
		return score;
	}*/

	int GetHDistance(Network *net)
	{
		ASSERT(net->GetnumVariables() == numVariables);

		BOOL * g1 = new BOOL[numVariables*numVariables];
		BOOL * g2 = new BOOL[numVariables*numVariables];
			
		GetStructure(g1);
		net->GetStructure(g2);

		int a=0;
		int d=0;
		int r=0;

		int i2, j2 ,k;

		for (int i = 0; i< numVariables; i++)
		{
			for (int j = 0; j < i; j++)
			{
				if (g1[i*numVariables+j] || g1[j*numVariables+i])
				{
					if (!g2[i*numVariables+j] && !g2[j*numVariables+i])
						a++;
					else
					{
						if (g1[i*numVariables+j] != g2[i*numVariables+j])
						{
							if (g1[i*numVariables+j])
							{
								i2 = i;
								j2 = j;
							}
							else
							{
								i2 = j;
								j2 = i;
							}
							
							//reversed edge makes graph equivalent ???
							//ie: is the edge i2->j2 in g1 involved in a immorality
							for ( k = 0; k < numVariables;k++)
							{
								if (g1[k*numVariables+j2])
								{
									//we have a V-structure i2->j2<-k
									//is is an immorality ?
									if (!g1[k*numVariables+i2] && !g1[i2*numVariables+k])
										break;
								}
							}

							if (k!=numVariables)
								r++;
						}
					}
				}
				else
				{
					if (g2[i*numVariables+j] || g2[j*numVariables+i])
						d++;
				}
			}
		}

		delete g1;
		delete g2;

		return a+d+r;

	}

	int GetMaxParent()
	{
		int mp = 0;
		for (int n = 0; n < numVariables; n++)
		{
			if (GetNode(n)->GetnumVariables()-1 > mp)
				mp = GetNode(n)->GetnumVariables()-1;
		}

		return mp;

	}

	int	GetNumEdges()
	{
		int ne = 0;
		for (int n = 0; n < numVariables; n++)
		{
			ne += GetNode(n)->GetnumVariables()-1;
		}

		return ne;
	}
}; // class Network



class RealNetwork : public Network
{

	char*	Name;
	char**	VarNames;
	char***	VarValues;


public:
	RealNetwork()
	{
		Name = NULL;
		VarNames = NULL;
		VarValues = NULL;
	}

	//Construct Bayesian Network from *.net file
        RealNetwork( char* filename, int type)
          {
            
            FILE *	f;
            char	buffer[BUFFER_SIZE];
            char	buffer2[BUFFER_SIZE];
            char 	sep[]=" (')\n\r	";
            char 	sep2[]=" '	";
            char 	sep2B[]="'()	\n\r";
            char 	sep3[]=" ()	\n\r";
            char	sepcsv[]=",";
            char	*s,*s2;
            int		n,d,v2,ii,j;
            int		*varvector;
            int		*assignment;
            float	ff;
            
            int i, v, k;
            
            Name = NULL;
            VarNames = NULL;
            VarValues = NULL;
            
            f = fopen(filename,"r");
            
            assert(f!=NULL);
            
            //fscanf( f ,"(network '%s :probability)\n",buffer);
            if (type == FULL)
              fscanf( f ,"(network %s :probability)\n",buffer);
            else
              strcpy(buffer,"real");

            if (type == CSV)
              {
		Name = new char[strlen(filename)+1];
		strcpy(Name, filename);
		fscanf( f ,"%s",buffer);
		
		n = 0;
		s = strtok(buffer, sepcsv);
		while (true)
		{
                  if (s == NULL)
                    break;
                  n++;
                  s = strtok(NULL, sepcsv);
                  
		}
                
		numVariables = n;
		Nodes = new Node[numVariables];
			
		VarNames = new char*[numVariables];
		VarValues = new char**[numVariables];
                
		fseek(f,SEEK_SET,0);
		
		fscanf( f ,"%s",buffer);
		
		n = 0;
		s = strtok(buffer, sepcsv);
		while (true)
                  {
                    if (s == NULL)
                      break;
                    
			VarNames[n] = new char[strlen(s)+1];
			strcpy(VarNames[n],s);
                        
			n++;
			s = strtok(NULL, sepcsv);
			
                  }
                
		
		for (i = 0 ; i < numVariables; i++)
                  {
                    Nodes[i].SetVariableDomain(0);
                    VarValues[i] = new char*[DOMAIN_MAX];
                  }
                
		fseek(f,SEEK_SET,0);
		fscanf( f ,"%s",buffer);
		//fscanf( f ,"%s",buffer);
                
		while (true)
                  {
                    fscanf( f ,"%s",buffer);
                    if (feof(f))
                      break;
                    
                    i = 0;
                    s = strtok(buffer, sepcsv);
                    while (true)
                      {
                        if (s == NULL)
                          break;
                        
                        //v = atoi( &s[1] ) - 1;
                        for (j = 0; j < Nodes[i].GetVariableDomain() ; j++)
                          {
                            if (strcmp(s, VarValues[i][j])==0)
                              break;
                          }
                        if (j == Nodes[i].GetVariableDomain())
                          {
                            VarValues[i][j] = new char[strlen(s)+1];
                            strcpy(VarValues[i][j],s);
                            Nodes[i].SetVariableDomain(j+1);
                            assert(j+1<=DOMAIN_MAX);
                          }
                        
                        s = strtok(NULL, sepcsv);
                        i++;
                        
                      }
                    assert( i == numVariables);
                  }
		
		return;
                
              }

            Name = new char[strlen(buffer)+1];
            strcpy(Name,buffer);
            
            //Count number of variables
            n = 0;
            
            while ( !feof(f) )
              {
                
		fgets( buffer, BUFFER_SIZE, f);
                
		if (feof(f))
                  break;
                
		if (sscanf(buffer, "(var '%s", buffer2) != 0 || sscanf(buffer, "(var %s", buffer2) != 0)
                  n++;
                
              }
            
            
            
            //Store variables name and size of domain
            numVariables = n;
            Nodes = new Node[numVariables];
            VarNames = new char*[numVariables];
            
            fseek(f,SEEK_SET,0);
            
            n = 0;
            
            while ( !feof(f) )
              {
                
		fgets( buffer, BUFFER_SIZE, f);
                
		if (feof(f))
                  break;
                
		s = strtok(buffer,sep);
		if (s!=NULL && !strcmp(s,"var"))
                  {
                    s = strtok(NULL,sep);
                    
                    VarNames[n] = new char[strlen(s)+1];
                    strcpy(VarNames[n],s);
                    
                    d=0;
                    s = strtok(NULL,sep);
                    while (s!=NULL)
                      {
                        d++;
                        s = strtok(NULL,sep);
                      } 
                    
                    Nodes[n].SetVariableDomain(d);
                    
                    n++;
                  }
                
		//fseek(f,SEEK_SET,0);
                
              }
            
            
            //read domain names
            VarValues = new char**[numVariables];
            
            fseek(f,SEEK_SET,0);
            
            n = 0;
            
            while ( !feof(f) )
              {
                
		fgets( buffer, BUFFER_SIZE, f);
		
		if (feof(f))
                  break;
		s = strtok(buffer,sep);
		if (s!=NULL && !strcmp(s,"var"))
                  {
                    s = strtok(NULL,sep);
                    
                    VarValues[n] = new char*[Nodes[n].GetVariableDomain()];
                    
                    d=0;
                    s = strtok(NULL,sep);
                    while (s!=NULL)
                      {
                        
                        VarValues[n][d] = new char[strlen(s)+1];
                        strcpy(VarValues[n][d],s);
                        
                        d++;
                        s = strtok(NULL,sep);
                      } 
                    
                    ASSERT(d == Nodes[n].GetVariableDomain());
                    
                    
                    n++;
                  }
                
                
              }
            
            if (type == NAME)
              return;
            
            //read number of parents

	fseek(f,SEEK_SET,0);


	while ( !feof(f) )
	{
		
		fgets( buffer, BUFFER_SIZE, f);

		s = strtok(buffer,sep);
		
		if (s!=NULL && !strcmp(s,"parents"))
		{
			s = strtok(NULL,sep2);//skip var name

			for (n = 0; n < numVariables; n++)
			{
				if (!strcmp(VarNames[n],s))
					break;
			}
			ASSERT(n != numVariables);

			d=1;

			s = strtok(NULL,sep2B);

			//ASSERT(s[0]=='(');
			
			s = strtok(s,sep3);
			
			while (s != NULL)
			{
				d++;
				s = strtok(NULL,sep3);
			} 

			Nodes[n].SetnumVariables(d);
			
		}


	}
	
	//read parents name and CPTs

	fseek(f,SEEK_SET,0);


	while ( !feof(f) )
	{
		
		fgets( buffer, BUFFER_SIZE, f);
		
		s = strtok(buffer,sep);

		if (s!=NULL && !strcmp(s,"parents"))
		{
			
			

			s = strtok(NULL,sep2);//skipping var name

			for (n = 0; n < numVariables; n++)
			{
				if (!strcmp(VarNames[n],s))
					break;
			}
			ASSERT(n != numVariables);

			//if (n==37)
			//	printf("???");

			varvector=new int[Nodes[n].GetnumVariables()];

			varvector[0] = n;

			s = strtok(NULL,sep2B);

			if (s!=NULL)
			{
				s2 = s + strlen(s) + 1;
			
				s = strtok(s,sep3);
			}
			else
			{
				s2 = NULL;

				ASSERT( Nodes[n].GetnumVariables() == 1 );
			}

			for (d = 1; d < Nodes[n].GetnumVariables(); d++)
			{

				//Search for corresponding variable number
				varvector[d] = UNDEF;

				for ( i = 0; i < numVariables ; i++)
				{

					ASSERT(s != NULL);

					if (!strcmp(VarNames[i],s))
						varvector[d] = i;
						
				}
				
				ASSERT( varvector[d] != UNDEF);

				s = strtok(NULL,sep3);

			} 
			
			Nodes[n].SetScope(varvector,this);

			d = Nodes[n].GetnumVariables();
			assignment = new int[d];

			//reading each value of the factor
			ASSERT( Nodes[n].GetSize() % Nodes[n].GetVariableDomain() == 0);

			ii = Nodes[n].GetSize()/Nodes[n].GetVariableDomain();

			if (s2!=NULL)
				s = strtok(s2,sep);
			else
				s = NULL;

			for ( i = 0; i<ii ; i++)
			{
				//fgets( buffer, BUFFER_SIZE, f);
				
				
				while (s == NULL)
				{
					fgets( buffer, BUFFER_SIZE, f);
					s = strtok(buffer,sep);
				}
				

				//building assignement vector
				for ( v = 1; v < d; v++)
				{

					assignment[v] = UNDEF;

					v2 = varvector[v];

					//looking for value of v2th variable
					
					for (  k = 0; k < Nodes[v2].GetVariableDomain(); k++)
					{
						if (!strcmp(VarValues[v2][k],s))
							assignment[v] = k;
					}

					if (assignment[v] == UNDEF)
						if (!OBS_SUPPRESS_OUTPUT) printf("ERROR");

					ASSERT(assignment[v] != UNDEF);

					s = strtok(NULL,sep);


				}
				//assignemnt built
		
				for ( k = 0; k < Nodes[n].GetVariableDomain(); k++)
				{
					
					assignment[0] = k;

					sscanf(s,"%f",&ff);

					Nodes[n].SetValue(ff, assignment);
						
					s = strtok(NULL,sep);
				}

				
			}
			
			delete varvector;
			delete assignment;

			ASSERT(Nodes[n].isValidCPT());

		}

	}

	for (n = 0; n < numVariables; n++)
		ASSERT(Nodes[n].isValidCPT());

	fclose(f);

          } //	RealNetwork( char* filename, int type);

	//Copy number of variables, domains and labels (without copying structure or CPTs)
	RealNetwork(RealNetwork *net)
	{
		
		int d,l;
		char * str;

		numVariables = net->GetnumVariables();
		Nodes = new Node[numVariables];
		VarNames = new char*[numVariables];
		VarValues = new char**[numVariables];

		str = net->GetName();
		l = strlen(str)+1;
		Name = new char[l];
		memcpy(Name,str,sizeof(char)*l);

		for (int i = 0; i < numVariables ;i++)
		{
			
			str = net->GetVariableName(i);
			l = strlen(str)+1;
			VarNames[i] = new char[l];
			memcpy(VarNames[i],str,sizeof(char)*l);
		
			d = net->GetNode(i)->GetVariableDomain();
			Nodes[i].SetVariableDomain(d);
			VarValues[i] = new char*[d];

			for (int j = 0; j < d ; j++)
			{
				str = net->GetVariableValue(i,j);
				l = strlen(str)+1;
				VarValues[i][j] = new char[l];
				memcpy(VarValues[i][j],str,sizeof(char)*l);
			}
		}
		
	}

	~RealNetwork()
	{

		for (int i = 0; i < numVariables ;i++)
		{
			for (int j = 0; j < Nodes[i].GetVariableDomain() ; j++)
				delete VarValues[i][j];

			delete VarValues[i];
			delete VarNames[i];
		}
		delete Name;
		delete VarNames;
		delete VarValues;

	}

	char* GetName()
	{
		return Name;
	}

	char* GetVariableName(int i)
	{
		return VarNames[i];
	}

	char* GetVariableValue(int v, int i)
	{
		return VarValues[v][i];
	}

	void ToFile(const char* filename )
	{
		FILE * f = fopen(filename,"w");

		if (f == NULL)
		{
			if (!OBS_SUPPRESS_OUTPUT) printf("Could not open file %s.\n",filename);
			ASSERT(FALSE);
		}

		int p, size,v ;

		fprintf(f, "(network %s :probability )\n", Name);

		fprintf(f, "\n");

		for (v = 0; v < numVariables; v++)
		{
			fprintf(f, "(var %s (", VarNames[v]);
			
			for (int d = 0; d < Nodes[v].GetVariableDomain() ; d++)
				fprintf(f, "%s ", VarValues[v][d]);

			fprintf(f,"))\n\n");

		}


		for (v = 0; v < numVariables; v++)
		{
			fprintf(f, "(parents %s ( ", VarNames[v]);
			
			int numv = Nodes[v].GetnumVariables();

			for (int d = 1; d < numv ; d++)
			{
				p = Nodes[v].GetVariable( d  );
				fprintf(f, "%s ", VarNames[p]);
			}

			fprintf(f,")");

			if (Nodes[v].GetnumVariables() != 1)
			{
				fprintf(f,"\n(\n");

				ASSERT(Nodes[v].GetSize() % Nodes[v].GetVariableDomain() == 0);

				size = Nodes[v].GetSize() / Nodes[v].GetVariableDomain();

				int * a = new int[numv];
				int * dom = new int[numv];
		
				Nodes[v].GetDomains(dom);

				for (p = 1; p < numv ; p++)
					a[p] = 0;

				for (int s = 0; s < size; s++)
				{
					fprintf(f,"   ((");
					
					for (p = 1; p < numv ; p++)
						fprintf(f,"  %s",VarValues[Nodes[v].GetVariable(p)][a[p]]);

					fprintf(f,") ");

					for (int i = 0; i < Nodes[v].GetVariableDomain(); i++)
					{
						a[0] = i;
						fprintf(f,"%f ",Nodes[v].GetValue(a));
					}

					NextAssignment(numv,a,dom);

					fprintf(f, ")\n");
				}

				delete a;
				delete dom;

				fprintf(f,"))\n\n");
			}
			else
			{
				fprintf(f," ( ");

				int * a = new int[numv];	
				
				for (int i = 0; i < Nodes[v].GetVariableDomain(); i++)
				{
					a[0] = i;
					fprintf(f,"%f ",Nodes[v].GetValue(a));
				}

				fprintf(f,"))\n\n");

				delete a;
			}


		}

		fclose(f);
	} // void ToFile( char * filename )

}; // class RealNetwork : public Network

/**
 * This class can read in and write a statistics file, which is essentially a
 * dataset.  Statistics files have the following format:
 *  - number of variables m (1 line)
 *  - domain sizes d_i (m lines)
 *    - (each variable's domain size on a different line)
 *  - number of samples n (1 line)
 *  - samples (nm lines)
 *    - (written sequentially, with each variable value on a new line)
 *  - variable marginals (sum_i d_i lines)
 *    - (for each variable, for each value, percent of samples with that value,
 *      one on each line)
 */
class Statistics {
  int	numVariables;
  
  int numSamples;
public:
  int	*Domains;
  double **mProb;
  int	*Samples;

  Statistics() {
    numVariables = 0;
    Domains = NULL;
    numSamples = 0;
    Samples = NULL;
  }

  Statistics(int nv, int *d, int ns, int *s) {
    numVariables = nv;
    Domains = new int[nv];
    memcpy(Domains, d, sizeof(int)*nv);
    
    numSamples = ns;
    Samples = new int[nv*ns];
    memcpy(Samples, s, sizeof(int)*ns*nv);
  }

  Statistics(Network* net, int nb) {
    int i,j,n;
    
    numVariables = net->GetnumVariables();
    numSamples = nb;

    Domains = new int[numVariables];
    Samples = new int[numVariables*numSamples];
    mProb = new double*[numVariables];
    
    net->GetDomains(Domains);

    for ( i = 0; i < numVariables;i++)
      {
        mProb[i] = new double[Domains[i]]; 
        for (int j = 0; j < Domains[i];j++)
          mProb[i][j] = STATS_DEFAULT;
      }

    int * ord = new int[numVariables];

    net->GetTopologicalOrdering(ord);
	
    for ( n = 0; n < nb ; n++ ) {
      net->ForwardSample( &Samples[n*numVariables], ord );
      for ( i = 0; i < numVariables;i++)
        mProb[i][Samples[n*numVariables+i]]++;
    }

    //Normalize PROB FOR BDE PRIOR
    for (i = 0; i < numVariables ; i++) {
      //if (i==16)
      //	printf("??");

      for ( j = 0; j < Domains[i]; j++) {
        //K2 PRIOR
#ifdef K2_PRIOR
        mProb[i][j] = 1;
#endif
        //BDE PRIOR
#ifdef BDE_PRIOR
        mProb[i][j] /= (double)nb + (double)Domains[i]*(double)STATS_DEFAULT;
#endif
      }
    }

    delete ord;
    
  }

  float GetAArity() {
    float res = 0;

    for (int n = 0; n < numVariables;n++)
      res+=Domains[n];

    return res/numVariables;
  }

  Statistics(char * filename) {
    int i,j;
    float d;

    FILE * f=fopen(filename,"r");
    ASSERT(f!=NULL);

    fscanf( f, "%d",&numVariables);

    Domains = new int[numVariables];
		
    for ( i = 0; i < numVariables;i++)
      fscanf( f, "%d",&Domains[i]);

    fscanf( f, "%d",&numSamples);

    Samples = new int[numVariables*numSamples];

    for ( i = 0; i < numSamples;i++)
      for ( j=0; j < numVariables; j++) {
        ASSERT( !feof(f) );
        fscanf( f, "%d",&Samples[i*numVariables+j]);
      }
				
    mProb = new double*[numVariables];
    
    for ( i = 0; i < numVariables;i++) {
      mProb[i] = new double[Domains[i]];
      for ( j=0; j < Domains[i]; j++) {
        ASSERT( !feof(f) );
        fscanf ( f, "%f",&d);
        //ASSERT( d!= 0);
        if(d == 0)
          d = 1e-7;
        mProb[i][j] = d;
      }
    }
    fclose(f);
  }

  Statistics(const prl::assignment_dataset& ds) {
    using namespace prl;
    numVariables = ds.num_variables();
    Domains = new int[numVariables];
    finite_var_vector varvec;
    foreach(finite_variable* v, ds.finite_list())
      varvec.push_back(v);
    for (int i = 0; i < numVariables; i++)
      Domains[i] = varvec[i]->size();
    numSamples = ds.size();
    Samples = new int[numVariables*numSamples];
    for (int i = 0; i < numSamples; i++) {
      std::vector<size_t> finite_val(ds[i].finite());
      for (int j = 0; j < numVariables; j++) {
        Samples[i * numVariables + j] = finite_val[j];
        if (finite_val[j] > ds.finite_list()[j]->size()) {
          assert(false);
        }
      }
    }
    mProb = new double*[numVariables];
    for (int i = 0; i < numVariables; i++) {
      mProb[i] = new double[Domains[i]];
      typedef table_factor factor_type;
      factor_type var_marginal = ds.marginal<factor_type>(varvec[i]);
      assignment a;
      for (int j = 0; j < Domains[i]; j++) {
        a.finite()[varvec[i]] = j;
//        a.finite()[varvec[i]] = finite_value(j);
        mProb[i][j] = std::max(1e-7, var_marginal(a));
      }
    }
  }

	Statistics(char * filename, RealNetwork *bn, int type)
	{
		int i,j,n;
		int d;
		char	buffer[BUFFER_SIZE];
		char	*s;

		FILE * f=fopen(filename,"r");
		ASSERT(f!=NULL);

		//fscanf( f, "%d",&numVariables);

		numVariables = bn->GetnumVariables();

		Domains = new int[numVariables];
		
		bn->GetDomains(Domains);
		//	fscanf( f, "%d",&Domains[i]);

		//fscanf( f, "%d",&numSamples);
		if (type == LIBB)
		{
			numSamples = 0;
			buffer[0] = 0;

			while (!feof(f))
			{
				if (buffer[0] != '(')
					fscanf(f,"%s",buffer);
		
				if (feof(f))
					break;

				for ( j=0; j < numVariables; j++)
				{
					fscanf(f,"%s",buffer);
				}
				fscanf(f,"%s",buffer);
				numSamples++;
			}

			fseek(f,SEEK_SET,0);

			Samples = new int[numVariables*numSamples];

			buffer[0] = 0;
			for ( i = 0; i < numSamples;i++)
			{
				if (buffer[0] != '(')
					fscanf(f,"%s",buffer);
				for ( j=0; j < numVariables; j++)
				{
					fscanf(f,"%s",buffer);
					s = strtok(buffer,")");
					for ( d = 0; d < Domains[j]; d++)
					{
						if (strcmp(s,bn->GetVariableValue(j,d))==0)
							break;
					}
					assert( d != Domains[j]);

					Samples[i*numVariables+j] = d;

				}

				fscanf(f,"%s",buffer);
			}
		}
		else
		{

			assert( type == CSV);
			numSamples = 0;

			fscanf(f,"%s",buffer);
			//fscanf(f,"%s",buffer);
			while (!feof(f))
			{
				fscanf(f,"%s",buffer);
		
				if (feof(f))
					break;

				numSamples++;
			}

			fseek(f,SEEK_SET,0);
			fscanf(f,"%s",buffer);
			//fscanf(f,"%s",buffer);

			Samples = new int[numVariables*numSamples];

			for ( i = 0; i < numSamples;i++)
			{

				fscanf(f,"%s",buffer);
				s = strtok(buffer,",");

				for ( j=0; j < numVariables; j++)
				{
					for ( d = 0; d < Domains[j]; d++)
					{
						if (strcmp(s,bn->GetVariableValue(j,d))==0)
							break;
					}
					assert( d != Domains[j]);

					Samples[i*numVariables+j] = d;
					s = strtok(NULL,",");

				}

			}


		}

		mProb = new double*[numVariables];
		
		for ( i = 0; i < numVariables;i++)
		{
			mProb[i] = new double[Domains[i]];
			for ( j = 0; j < Domains[i];j++)
				mProb[i][j] = STATS_DEFAULT;
		}

		
		for ( n = 0; n < numSamples ; n++ )
		{
			for ( i = 0; i < numVariables;i++)
				mProb[i][Samples[n*numVariables+i]]++;
		}


		//Normalize PROB FOR BDE PRIOR
		for (i = 0; i < numVariables ; i++)
		{
			//if (i==16)
			//	printf("??");

			for ( j = 0; j < Domains[i]; j++)
			{
				//K2 PRIOR
#ifdef K2_PRIOR
				mProb[i][j] = 1;
#endif
				//BDE PRIOR
#ifdef BDE_PRIOR
				mProb[i][j] /= (double)numSamples + (double)Domains[i]*(double)STATS_DEFAULT;
#endif
			}
		}

		fclose(f);
	}

	void toFile(const char* filename)
	{
		
		int i,j;

		FILE * f=fopen(filename,"w");
		
		fprintf( f, "%d\n",numVariables);
	
		for ( i = 0; i < numVariables;i++)
			fprintf( f, "%d\n",Domains[i]);

		fprintf( f, "%d\n",numSamples);

		for ( i = 0; i < numSamples;i++)
			for ( j=0; j < numVariables; j++)
				fprintf( f, "%d\n",Samples[i*numVariables+j]);
						
		for ( i = 0; i < numVariables;i++)
		{
			for ( j=0; j < Domains[i]; j++)
			{
				ASSERT( ((float) mProb[i][j]) != (float) 0.0 );
				//char test[256];
				//sprintf(test,"%.8f",mProb[i][j]);
				fprintf ( f, "%.8f\n",mProb[i][j]);
			}
		}
		fclose(f);

	}
	~Statistics()
	{	
		delete	Domains;
		delete	Samples;	
		for (int i = 0; i < numVariables;i++)
			delete	mProb[i];
		
		delete	mProb;
	}

	int GetnumVariables(){ return numVariables; };
	int GetVariableDomain(int v){ return Domains[v]; };
	int GetnumSamples(){ return numSamples; };
};

class ADTree;

struct VaryNode
{
	char	MCV;
	char	numTree;
	ADTree		**ADTrees;	
};

static		int*  poolCount;
static		int** poolRecords;
//static		ADTree** poolTree;

static		int	poolCountPtr;
static		int poolRecordsPtr;
//static		int poolTreePtr;

class ADTree
{
	//int			depth;

	unsigned short int var;
	unsigned short int numVary;
	int			count;
	VaryNode	*VaryNodes;
	//int			numLeaves;
	int			*Leaves;
	int			Rmin;

public:

	//prendre un tableau de pointeur plutot qu'un tableau d'entiers
	//meilleur pour leaf node ???
	ADTree(int n, int *Records, int numRecords, Statistics * stats, int depth, int rmin)
	{

		ASSERT(stats->GetnumVariables() < 65535);

		Rmin = rmin;
		var = n;
		count = numRecords;
		if (depth!=0)
		{
			if (numRecords > Rmin)
			{
				Leaves = NULL;
				numVary = stats->GetnumVariables() - n;
				VaryNodes = new VaryNode[stats->GetnumVariables() - n];
				for (int j = 0; j < numVary; j++)
					MakeVaryNode(&VaryNodes[j], j + n, Records, numRecords, stats, depth);
			}
			else
			{
				numVary = 0;
				VaryNodes = NULL;
				//numLeaves = numRecords;
				Leaves= new int[numRecords];
				memcpy(Leaves, Records, sizeof(int)*numRecords);
			}

			//DEBUG
			/*int c;
			for (int i = 0; i < numVary; i++)
			{
				c = 0;

				for (int k = 0; k < VaryNodes[i].numTree; k++)
					if (VaryNodes[i].ADTrees[k] != NULL)
						c+= VaryNodes[i].ADTrees[k]->count;

				if (c != count)
					printf("ERROR");
			}*/
		}
		else
		{
			numVary = 0;
			Leaves = NULL;
			VaryNodes = NULL;
		}
	};

	~ADTree()
	{
	
		for (int j = 0; j < numVary; j++)
		{
			for (int k = 0; k < VaryNodes[j].numTree; k++)
			{
				delete VaryNodes[j].ADTrees[k];
			}
			delete[] VaryNodes[j].ADTrees;
		}

		delete[] Leaves;
		delete[] VaryNodes;

	};

	
	void MakeVaryNode(VaryNode * vn, int n, int * Records, int numRecords, Statistics * stats, int depth)
	{
		ASSERT(stats->GetVariableDomain(n) < 256);

		int d = stats->GetVariableDomain(n);

		int ** ChildRecords = &poolRecords[ poolRecordsPtr ];//= new int*[ d ];//*numRecords ];
		int * ChildnumRecords = &poolCount[ poolCountPtr ]; //new int[ d ];
		
		poolRecordsPtr += d;
		poolCountPtr += d;


		vn->ADTrees = new ADTree*[ d ];
		vn->numTree = d;

		int * ptr = stats->Samples + n ;
		int nv = stats->GetnumVariables();

		int j,k;

		for ( k = 0 ; k < d ; k++ )
		{
			//ChildRecords[k] = new int[ numRecords ];
			ChildnumRecords[k] = 0;
		}

		for ( j = 0 ; j < numRecords ; j++)
		{
			k = *(ptr + Records[j]*nv);//ptr ;
			ChildRecords[ k ] [ ChildnumRecords[k]++ ] = Records[j];	
			//ptr += nv;
		}
	
		int max = 0;

		for ( k = 0 ; k < d ; k++ )
		{
			if ( ChildnumRecords[ k ] > max)
			{
				max = ChildnumRecords[ k ] ;
				vn->MCV = k;
			}
		}

		ASSERT( max != 0 );

		for ( k = 0 ; k < d ; k++ )
		{

			if (ChildnumRecords[ k ] == 0 || k == vn->MCV )
				vn->ADTrees[k] = NULL;
			else
				vn->ADTrees[k] = new ADTree( n+1, ChildRecords[ k ], ChildnumRecords[ k ], stats, depth - 1, Rmin);	
			
		}

		
		//for ( k = 0 ; k < d ; k++ )
		//	delete[] ChildRecords[k];

		//delete[] ChildnumRecords;
		//delete[] ChildRecords;
		poolRecordsPtr -= d;
		poolCountPtr -= d;
	}

	void MakeContab(int numVariables, int *Variables, int * dest, int size, Statistics *stats)
	{

		if (numVariables == 0)
		{
			ASSERT(size == 1);
			dest[0] = count;
			return;
		}

		int k,p,a;
		int *sample;
		int *ptr = dest;
		int *dom;
		//int* dest2;
		if (count <= Rmin)
		{
			//dest2 = new int[size];
			ptr = dest;

			dom = stats->Domains;
			int nv = stats->GetnumVariables();

			for (p = 0; p < size; p++)
				*(ptr++) = 0;

			for (p = 0; p < count; p++)
			{
				sample = stats->Samples + nv*Leaves[p];
				a = sample[Variables[0]];

				for (k = 1; k < numVariables; k++)
				{
					a *= dom[Variables[k]];
					a += sample[Variables[k]];
				}

				//if (a <0 || a>=size)
				//	printf("ERROR");

				dest[a]++;
				
			}

			return;
		}


		VaryNode *vn = &VaryNodes[ Variables[0] - var ];
		int d = vn->numTree;
		int s = (size / d);
		int m = vn->MCV;
		int *ptr2;
		int *ptr3;

		//ptr = dest;

		for (k = 0; k < d ; k++)
		{
			if (k != m)
			{
				if ( vn->ADTrees[k] != NULL)
					vn->ADTrees[k]->MakeContab(numVariables - 1, Variables + 1, ptr, s, stats);	
				else
				{
					ptr2 = ptr;
					for (p = 0; p < s; p++)
						*(ptr2++) = 0;
				}
			}
			else
				MakeContab(numVariables - 1, Variables + 1, ptr, s, stats);

			ptr += s;
		}

		
		ptr = dest;
		ptr3 = dest+s*m;

		for (k = 0; k < d ; k++)
		{
			if (k != m)
			{
				ptr2 = ptr3;
				for (p = 0; p < s; p++)
				{
					//if (dest[s*m+p] < dest[s*k+p])
					//	printf("ERORO");
					*(ptr2++) -= *(ptr++);//dest[s*m+p] -= dest[s*k+p];//;*(ptr++);
				}
			}
			else
				ptr+=s;
		}

		/*if (numLeaves != 0)
		{
			for (p=0; p < size; p++)
				if (dest[p]!=dest2[p])
					printf("ERROR");

			delete dest2;
		}
		
		int * dest3 = new int[s];
		vn->ADTrees[m]->MakeContab(numVariables - 1, Variables + 1, dest3, s, stats);
		for (p = 0; p < s; p++)
		{
			if (dest[s*m+p] != dest3[p])
				printf("ERIR");
		}
		delete dest3;*/
	}

};

class Counter
{
	int		numVariables;
	int*	Variables;
	int*	Domains;
	//int		numSamples;

private:
	int**	coord;//coord[i,j] gives the pointer add of the ith variable having jth value
	int		tablesize;
public:
	int*	table;//the count of each instance
	double* virtualtable;//the probanilities of each instance
	
	double		numVirtualSamples;//of BDE prior
public:

	Counter()
	{
		numVariables=0;
		Variables = NULL;
		Domains = NULL;
		table = NULL;
		virtualtable = NULL;
		coord=NULL;
	}

	//prob is the table of marginal probability of each variable
	void InitCounter(int nvars, int* vars, int* domains, Statistics *stats, double virtSampleSize)
	{
		int i,j,s;

		for ( i=0; i < numVariables; i++)
			delete coord[i];

		delete coord;
		delete Variables;
		delete Domains;
		delete table;
		delete virtualtable;

		int *a,*pSamples;

		numVirtualSamples = virtSampleSize;
		//numSamples = stats->GetnumSamples();

		numVariables = nvars;

		Variables =  new int[ numVariables ];
		Domains = new int[ numVariables ];
		coord = new int *[ numVariables ];

		memcpy(Variables,vars,sizeof(int)*numVariables);
		memcpy(Domains,domains,sizeof(int)*numVariables);

		tablesize = 1;
		//int step = 1;
		for ( i = numVariables-1 ; i >=0  ; i--)
		//for ( i = 0; i < numVariables  ; i++)
		{
			coord[i] = new int[Domains[i]];

			
			for ( j = 0; j < Domains[i] ; j++)
				coord[i][j] = tablesize * j;

			//step *= Domains[i];
	
			tablesize *= Domains[i];

		}
		table = new int[tablesize];
		virtualtable = new double[tablesize];

		for ( i = 0 ; i < tablesize ; i++ )
			table[i] = 0;

		pSamples = stats->Samples;

		int ns = stats->GetnumSamples();
		int nv = stats->GetnumVariables();
		
		a = new int[numVariables];

		int * ptr = table;

		for ( s = 0 ; s < ns ; s++)
		{	
			ptr = table;

			for ( i = 0; i < numVariables; i++)
			{
				ptr += coord[i][ pSamples[Variables[i]] ];
			}
		
			(*ptr)++;

			pSamples += nv ;
		}

		//We populate the virtualtable
		double p;

		for (i = 0; i < numVariables ; i++)
			a[i]=0;

		double * vptr = virtualtable;
		for (i = 0; i < tablesize ; i++)
		{
			p = 1;

			for ( j = numVariables-1 ; j >=0  ; j--)
			//for ( j = 0; j < numVariables ; j++)
				p *= stats->mProb[Variables[j]][a[j]];

			if (p==0.0)
				if (!OBS_SUPPRESS_OUTPUT) printf("ERROR");

			*(vptr++) = p;

			//NextAssignment(numVariables,a,Domains);
			NextAssignmentRight(numVariables,a,Domains);

		}
				

		delete a;

	}



	//prob is the table of marginal probability of each variable
	void InitCounter(int nvars, int* vars, int* domains, Statistics *stats, double virtSampleSize, ADTree * tree)
	{
		int i,j;
		int *a;
		double p;
		double * vptr;
		int *aptr, *vaptr, *dptr;

		delete table;
		delete virtualtable;

		numVirtualSamples = virtSampleSize;
		//numSamples = stats->GetnumSamples();
		

		a = new int[nvars];
		
		tablesize = 1;
		dptr = domains;
		aptr = a;
		for ( i = 0; i < nvars; i++)
		{
			tablesize *= *(dptr++);
			*(aptr++) =0;
		}

		table = new int[tablesize];
		virtualtable = new double[tablesize];

		tree->MakeContab(nvars, vars, table, tablesize, stats);

		vptr = virtualtable;
		for (i = 0; i < tablesize ; i++)
		{
			p = 1;

			aptr = a;
			vaptr = vars;

			for ( j = 0; j < nvars ; j++)
				p *= stats->mProb[ *(vaptr++) ][ *(aptr++) ];

			*(vptr++) = p;

			NextAssignmentRight(nvars,a,domains);

		}
				

		delete a;

	}

	Counter(int nvars, int* vars, int* domains, Statistics *stats, double initVal)
	{
		numVariables = 0;
		Variables = NULL;
		Domains = NULL;
		table = NULL;
		virtualtable = NULL;
		coord= NULL;

		InitCounter(nvars, vars, domains, stats, initVal);
	}

	Counter(int nvars, int* vars, Network *net, Statistics *stats, double initVal)
	{
		numVariables = 0;
		Variables = NULL;
		Domains = NULL;
		table = NULL;
		virtualtable = NULL;
		coord= NULL;

		ASSERT( net->GetnumVariables() == stats-> GetnumVariables() );
	
		int * dom = new int[nvars];

		for (int i = 0; i < nvars; i++)
		{
			ASSERT( net->GetNode(i)->GetVariableDomain() == stats->GetVariableDomain(i) );
			dom[i] = stats->GetVariableDomain(vars[i]);
		}

		InitCounter(nvars, vars, dom, stats, initVal);

		delete dom;
	}
	
	Counter(Node *node, Statistics *stats, double initVal)
	{
		numVariables = 0;
		Variables = NULL;
		Domains = NULL;
		table = NULL;
		virtualtable = NULL;
		coord= NULL;

		int	nvars = node->GetnumVariables();
		int * dom = new int[nvars];
		int * vars = new int[nvars];

		node->GetScope(vars);
		node->GetDomains(dom);

		InitCounter(nvars, vars, dom, stats, initVal);

		delete dom;
		delete vars;

	}

	~Counter()
	{
		for (int i=0; i < numVariables; i++)
			delete coord[i];

		delete coord;
		delete	Variables;
		delete	Domains;
		delete	table;
		delete  virtualtable;
	}

	int		GetPos(int* assignment)
	{
	
		//if (numVariables == 0)
		//	return 0;

		int p=0;//assignment[numVariables-1];

		/*for (int i = numVariables - 2 ; i >= 0 ; i--)
		{

			p *= Domains[i];
			p += assignment[i];
				
		}

		int p2=0;*/
		for (int i = 0 ; i < numVariables ; i++)
			p += coord[i][assignment[i]];
		//ASSERT(p == p2);
		//ASSERT( p < tablesize );
		return p;
	}

	//void AddCount(int * assignment){ table[GetPos(assignment)]++; }
	void AddCount(int * assignment)
	{
		int *ptr = table;
		for (int i = 0 ; i < numVariables ; i++)
			ptr += coord[i][assignment[i]];
		(*ptr)++;
	
	}

	double GetRealCount(int * assignment){	return table[GetPos(assignment)]; }
	double GetCount(int * assignment)
	{
		int p = GetPos(assignment); 
		return numVirtualSamples*virtualtable[p] + table[p]; 
	}

	double GetLikelihoodValue(Counter &trainingCounter)
	{
		
    int *rTTrain = trainingCounter.table;
    int total_samples = 0;
    for (int q = 0 ; q < tablesize; q++)
      total_samples += *(rTTrain++);

    double REGULARIZATION_SAMPLES = 1;

		//OPTIMIZATION !!!!
		double	value=0;
		double  rval, rvalTrain;
		double	vval;
		//int		ptr;
		//int		v,k;

		/*int * as = new int[numVariables];
		for ( k = 0; k < numVariables; k++)
			as[k] = 0;

		int m;*/


		//iterates through assignement
		int *rT = table;
    rTTrain = trainingCounter.table;
		double *vT = virtualtable;

		for (int p = 0 ; p < tablesize; p++)
		{
			rval = *(rT++);
      rvalTrain = *(rTTrain++) + (REGULARIZATION_SAMPLES / tablesize);
      //rvalTrain = rval;
			vval = numVirtualSamples * (*(vT++));
			//score += -dlgamma(vval)+dlgamma(rval+vval);
			//printf("%f %f\n",rval,vval);
//			if (rvalTrain != 0)
//				value += (rval)*log(rvalTrain);
      if (rval != 0)
				value += (rval)*log(rvalTrain);
      //else if (rval != 0)
      //{
      //  value += (rval)*MINUS_INFINITY;
      //  break;
      //}

		}
	
		return value;

	}

	double GetLikelihoodScore(int var)
	{
		//return the log likelihood score
		//OPTIMIZATION !!!!
		double	score=0;
		double  rcount,vcount;
		double	rval,vval;
		int		ptr,v,k;

		int * as = new int[numVariables];
		for ( k = 0; k < numVariables; k++)
			as[k] = 0;

		//we sum out the current variable
		for (  v = 0; v < numVariables; v++)
			if ( var == Variables[v] )
				break;

		ASSERT(v != numVariables );

		int	d = Domains[v];
		int m;

		//iterates through assignement
		do
		{
			vcount = 0;
			rcount = 0;

			for (k = 0 ; k < d; k++)
			{
				as[v] = k;
				ptr = GetPos(as);
				
				rval = table[ptr];
				vval = numVirtualSamples * virtualtable[ptr];
				
				rcount += rval;
				vcount += vval;

				score += (rval+vval)*log(rval+vval);
			}

			score -= (rcount+vcount)*log(rcount+vcount);

			m = NextAssignment(numVariables, as, Domains);
		

		} while (m < numVariables);

		delete	as;

		return score;

	}

	/*double GetBICScore(int var)
	{
		double score = GetLikelihoodScore(var);

		int dim = 1;
		for ( int v = 0; v < numVariables; v++)
			if ( var != Variables[v] )
				dim *= Domains[v];
			else
				dim *= Domains[v] - 1;
		
		return score - (log(numVirtualSamples+numSamples)/(double)2.0)*((double)dim);

	}*/

	double GetBayesianScore(int var)
	{
		
		//OPTIMIZATION !!!!
		double	score=0;

		//DEBUG
		double score2 = 0;
		double score3 = 0;

		double  rcount,vcount;
		double	rval,vval;
		int		ptr,v,k;

		int * as = new int[numVariables];
		for ( k = 0; k < numVariables; k++)
			as[k] = 0;

		//we sum out the current variable
		//printf("Bayesian of %d [", var);
		for (  v = 0; v < numVariables; v++)
			if ( var == Variables[v] )
				break;
			//else
			//	printf("%d,",Variables[v]);

		//printf("]\n");

		ASSERT(v != numVariables );

		int	d = Domains[v];
		int m;


		//iterates through assignement
		do
		{
			vcount = 0;
			rcount = 0;

			//printf("VAL ");
			for (k = 0 ; k < d; k++)
			{
				as[v] = k;
				ptr = GetPos(as);
				
				rval = table[ptr];
				vval = numVirtualSamples * virtualtable[ptr];
				
				rcount += rval;
				vcount += vval;

				if (vval!=0.0)
				score += -dlgamma(vval)+dlgamma(rval+vval);
				else
					if (!OBS_SUPPRESS_OUTPUT) printf("!");

				//printf("%f %f | ",rval,vval);
				score2 += -dlgamma(vval)+dlgamma(rval+vval);
			}

			//printf("\n");

			//printf("COUNT %f %f\n",rcount,vcount);

			if (vcount!=0.0)
			score += dlgamma(vcount)-dlgamma(rcount+vcount);
			else
				if (!OBS_SUPPRESS_OUTPUT) printf("!");

			score3 += dlgamma(vcount)-dlgamma(rcount+vcount);

		//	m = NextAssignment(numVariables, as, Domains);
			m = NextAssignmentRight(numVariables, as, Domains);
		

		} while (m !=-1);//< numVariables);

		delete[] as;

		//printf("%f %f\n", score2, score3);

		return score;

	}

	double GetScore(int var)
	{
//#ifdef BAYESCORE
		return GetBayesianScore(var);
/*#endif
#ifdef BICSCORE
		return GetBICScore(var);
#endif*/
	}

	double GetBayesianValue()
	{
		
		//OPTIMIZATION !!!!
		double	score=0;
		double  rval,vval;
		//int		ptr;
		//int		v,k;

		/*int * as = new int[numVariables];
		for ( k = 0; k < numVariables; k++)
			as[k] = 0;

		int m;*/


		//iterates through assignement
		int *rT = table;
		double *vT = virtualtable;

		for (int p = 0 ; p < tablesize; p++)
		{
			rval = *(rT++);
			vval = numVirtualSamples * (*(vT++));
			score += -dlgamma(vval)+dlgamma(rval+vval);
			//printf("%f %f\n",rval,vval);

		}
		/*
		do
		{
			ptr = GetPos(as);
			
			rval = table[ptr];
			vval = numVirtualSamples * virtualtable[ptr];
		
			//printf("%f %f\n",rval,vval);

			if (vval!=0.0)
				score += -dlgamma(vval)+dlgamma(rval+vval);
			else
				printf("!");
			
			m = NextAssignmentRight(numVariables, as, Domains);
		
		} while (m !=-1);//< numVariables);

		delete[] as;
		*/
		return score;

	}
};




/*class FamScoreNode
{
public:
	double			Score;
	double			maxScore;//maximum scre of descendants nodes in the tree
	
	//int				Variable;//Xi
	//int				numParents;//=depth of the node in the tree >=0 <=nummaxparents
	//int				*Parents;//Pa(Xi)  Parents[i] = gives the ith parent

	//int				numNextParent;//=num of different parents we can add = n - 1 - numParents except if we have reached max parents
	//int				*NextParent;//NextParent[i] = gives the ith parent variable we can add

	//FamScoreNode	**NextScores;//Given in the coordinate of variables whithout the root ! 
								 //so is valid only for numVariables - 1 - numParents values !
	
public:
	FamScoreNode()
	{
		//DEBUG
		Score = 1;

		//Children = NULL;
		//ChildrenScore = NULL;
	}

	FamScoreNode(int ptr, int numVariables, int numparents, int maxParents)
	{
		//Parents = NULL;
		//NextParent = NULL;
		//NextScores = NULL;
		//Init(ptr, numVariables, numParents, maxParents);
	}

	//ptr is the pointer of this node in the node table
	void Init(int ptr, int numVariables, int numparents, int maxParents)
	{
		//numParents = numparents;
		//Parents = new int[numParents];
		//PtrToSet(ptr, numVariables - 1, numParents, Parents);
		//if ( numParents < maxParents)
		{
			//numNextParent = numVariables - 1;
			//NextParent = new int[numNextParent];
			//NextScores = new FamScoreNode*[numVariables - 1];
		}
		//else
		{
			//numNextParent = 0;
			//NextParent = NULL;
			//NextScores = NULL;
		}
	}

	~FamScoreNode()
	{
		//delete	Parents;
		//delete	NextParent;
		//delete	NextScores;
	}

};
*/

struct Family
{
	double  score;			//score of the family
	//int		validCount;		//num of parents
#ifdef ORDER_SPACE
	double	likelihood;			//loglikelihood
	int		numParents;		//num of parents
	int		ptr;			//order (so that we can have its componenets)
#endif
	//int		*Parents;
};


#ifdef GRAPH_SPACE
#define NB_TYPE_OPERATOR 3
#define ADD_EDGE 0
#define DELETE_EDGE 1
#define REVERSE_EDGE 2
#endif


#ifdef ORDER_SPACE
#define NB_TYPE_OPERATOR 1
#define REVERSE_NODE 0
#endif

struct Operator
{


#ifdef GRAPH_SPACE
	int type;
	int from;
	int	to;
	int cand;
	int icand;

	BOOL equalTo(Operator o)
	{
		//return from == o.from && to == o.to && type == o.type;
		return cand == o.cand && to == o.to && type == o.type;
	}

	BOOL isReverseOf(Operator o)
	{
		if (o.type == UNDEF)
			return FALSE;

		if (o.type == type && o.from == from && o.to == to)
			return TRUE;

		switch (type)
		{
		case ADD_EDGE:
			if (o.type == DELETE_EDGE && o.to == to && o.from == from)
				return TRUE;
			return FALSE;
		case DELETE_EDGE:
			if (o.type == ADD_EDGE && o.to == to && o.from == from)
				return TRUE;
			return FALSE;
		case REVERSE_EDGE:
			if ((o.type == REVERSE_EDGE || o.type == ADD_EDGE) && o.to == from && o.from == to)
				return TRUE;
			return FALSE;
		}
		return FALSE;
	}

#endif


#ifdef ORDER_SPACE
	int type;
	int pos;
	int	var1;
	int var2;

	BOOL isReverseOf(Operator o)
	{
		return (o.var1 == var2 && o.var2 == var1 && type==o.type);
	}


	BOOL isEqualTo(Operator o)
	{
		return (o.var1 == var1 && o.var2 == var2 && type==o.type);
	}


	BOOL equalTo(Operator o)
	{
		return (o.var1 == var1 && o.var2 == var2 && o.pos == pos && type==o.type);
	}
#endif
};


class State;

struct ElemParent
{
	int		val;
	int		Previous;
	int		Next;
};

class ListParent
{
	int		FirstParent;
	int		LastParent;
	int		numParents;
	int		numVariables;
	ElemParent* List;  //sorted list of parents
	int		*ParentPos;//position of parent in the list

public:
	ListParent(){};

	void Init(int n)
	{
		List = new ElemParent[n];
		ParentPos = new int[n];

		for (int i = 0 ; i < n ; i++)
		{
			List[i].Previous = i-1;
			List[i].Next = i+1;
		}
		List[0].Previous = n-1;
		List[n-1].Next = 0;

		FirstParent = 0;
		LastParent = 0;
		numParents = 0;
		numVariables = n;
	}

	void CopyFrom( ListParent *l)
	{
		assert(numVariables == l->numVariables );

		memcpy( List, l->List, sizeof(ElemParent) * numVariables);
		memcpy(ParentPos, l->ParentPos, sizeof(int) * numVariables);

		FirstParent = l->FirstParent ;
		LastParent = l->LastParent ;
		numParents = l->numParents ;

	}

	~ListParent()
	{
		delete List;
		delete ParentPos;
	}

	void Push(int p)
	{
		//isValid();
		assert(p!=UNDEF);

		int pos = LastParent;

		ParentPos[p] = pos;
		List[pos].val = p;

		LastParent = List[pos].Next;
		numParents++;

		if (numParents == 1)
		{
			FirstParent = pos;
			return;
		}

		int p0 = List[pos].Previous;
		int p1 = List[pos].Next;
		
		List[p1].Previous = p0;
		List[p0].Next = p1;

		int p2 = FirstParent;

		while (true)
		{
			if (List[p2].val > p)
				break;
			
			p2 = List[p2].Next;

			if (p2 == LastParent)
				break;

		}
		//we must insert p before p2
		p0 = List[p2].Previous;

		List[p2].Previous = pos;
		List[p0].Next = pos;
		List[pos].Next = p2;
		List[pos].Previous = p0;

		if (p2 == FirstParent)
			FirstParent = pos;

		//isValid();
	}

	void	Pop(int p)//pop the parent p
	{
		numParents--;

		int p1 = ParentPos[p];
		
		int p0 = List[p1].Previous;
		int p2 = List[p1].Next;

		List[p0].Next = p2;
		List[p2].Previous = p0;

		int l0 = List[LastParent].Previous;
		int l2 = LastParent;

		List[l0].Next = p1;
		List[p1].Previous = l0;
		
		List[p1].Next = l2;
		List[l2].Previous = p1;

		LastParent = p1;

		if (p1 == FirstParent)
			FirstParent = p2;

		if (numParents == 0)
			LastParent = p2;

		//isValid();
	}

	int GetNumParent()
	{
		return numParents;
	}

	ElemParent*	GetFirstParent()
	{
		return &List[FirstParent];
	}

	ElemParent* GetNextParent(ElemParent * l)
	{
		return &List[l->Next];
	}

	/*bool isValid()
	{
		int p = 0;
		int p1,v;

		for ( int i = 0 ; i < numVariables; i++)
		{
			p1 = List[p].Next;
			if (List[p1].Previous != p)
				return false;
			p = p1;
		}

		if ( p != 0)
			return false;

		if (numParents==0)
			return true;

		p = FirstParent;
		v = List[p].val;
		if (ParentPos[v] != p)
			return false;

		for ( i = 1; i < numParents; i++)
		{
			p1 = List[p].Next;

			if (List[p1].val <= v)
				return false;

			if (ParentPos[ List[p1].val ] != p1)
				return false;
		
			p = p1;
		}

		return true;
	}*/
};

struct Chain
{
	double		Value;
	Operator	Op;
	int			Previous;
	int			Next;
	int			Id;
	bool		Valid;
	int			Tabu;
	
	bool		Possible; //=Valid && Tabu == 0
};



class ListOp
{
private:
	int		FirstPossible;
	int		LastPossible;
	int		numPossible;
	int		numTotal;

	Chain	*List;

	void	SetPossible(int i)
	{
		SetScore(i, List[i].Value);
	}

	void	SetImpossible(int i)
	{
		List[i].Possible = false;

		int p0 = List[i].Previous;
		int p2 = List[i].Next;

		List[p0].Next = p2;
		List[p2].Previous = p0;

		int l0 = List[LastPossible].Previous;
		int l2 = LastPossible;

		List[l0].Next = i;
		List[i].Previous = l0;
		
		List[i].Next = l2;
		List[l2].Previous = i;

		LastPossible = i;

		if (i == FirstPossible)
			FirstPossible = p2;

		numPossible--;


	}
	

public:

	void CopyAllExceptTabuFrom(ListOp *l)
	{
		int i;
		//assert(isValid());

		for (i = 0; i < numTotal; i++)
		{
			List[i].Valid = List[i].Possible = false;
		}

		FirstPossible = LastPossible = 0;
		numPossible = 0;

		for (i = 0; i < numTotal; i++)
		{
			
			List[i].Value = l->List[i].Value;
			List[i].Valid = l->List[i].Valid;

			if (List[i].Valid)
				SetPossible(i);
			
		}

		//assert(isValid());

	}

	ListOp(State *s);

	virtual ~ListOp()
	{
		delete List;
	}


	void    SetInvalid(int i)
	{
		if (List[i].Valid == true && List[i].Tabu == 0)
			SetImpossible(i);

		List[i].Valid = false;		
	
		//assert(isValid());

	}

	void	SetScore(int i, double v)
	{

		//if (List[i].Op.type  == 2 && 
		//		( (List[i].Op.from==2 && List[i].Op.to==21) || (List[i].Op.from==8 && List[i].Op.to==4) ) )
		//		printf("??");

		//assert(v!=0);

		List[i].Value = v;

		List[i].Valid = true;

		if (List[i].Tabu != 0)
			return;

		if (numPossible == 0)
		{

			FirstPossible = i;
			LastPossible = List[i].Next;

			numPossible++;
			List[i].Possible = true;

			//assert(isValid());
			return;
		}
		
		if (List[i].Possible && i == FirstPossible)
		{
			FirstPossible = List[i].Next;
		}

		if (!List[i].Possible)
		{
			numPossible++;
			List[i].Possible = true;
		}
		
		
		int i0 = List[i].Previous;
		int i1 = List[i].Next;
		
		List[i0].Next = i1;
		List[i1].Previous = i0;

		if (LastPossible == i)
			LastPossible = i1;

		
		int p = FirstPossible;
		
		while (TRUE)
		{
			if (List[p].Value <= v )
				break;

			p = List[p].Next;

			if (p == LastPossible)
				break;
		}


		//i should be put before p
		int p0 = List[p].Previous;

		if (p == LastPossible)
		{
		
			List[p0].Next = i;
			List[i].Previous = p0;
			List[i].Next = p;
			List[p].Previous = i;
			//assert(isValid());
			return;			
		}

		int c = 1;

		while (List[p].Value == v) 
		//	&& (List[p].Op.type < List[i].Op.type || 
		//		( List[p].Op.type == List[i].Op.type && ( List[p].Op.from < List[i].Op.from || 
		//		(  List[p].Op.from == List[i].Op.from && List[p].Op.to < List[i].Op.to)))))
		{
			c++;

			p = List[p].Next;

			if (p == LastPossible)
				break;
		}
		
	//	printf("c=%d\n",c);
	//	double d = lrand48();
	//	c = (int) (d*c/(RAND_MAX+1));
	//	printf("d=%f c=%d\n",d, c);

		c = (int) (RANDOM*(double)c);
		
		//assert(c!=-1);

		while (c!=0)
		{
			p0 = List[p0].Next;
			c--;
		}

		//will be put after p0
		p = List[p0].Next;

		List[p0].Next = i;
		List[i].Previous = p0;
		List[i].Next = p;
		List[p].Previous = i;

		if (p == FirstPossible)
			FirstPossible = i;
	
		//assert(isValid());

	}

	void	IncTabu(int i, int v)
	{
		List[i].Tabu += v;

		if (v == -1)
		{
			if (List[i].Tabu == 0 && List[i].Valid)
				SetPossible(i);
		}
		else if (v == 1)
		{
			if (List[i].Tabu == 1 && List[i].Valid)
				SetImpossible(i);
		}

		
	//	assert(isValid());


	}

	bool isValid(int i)
	{
		return List[i].Valid;
	}

	double GetScore(int i)
	{
		assert( List[i].Valid );
		return List[i].Value;
	}

	Chain*	GetFirst()
	{
		return &List[FirstPossible];
	}

	Chain*	GetNext(Chain *c)
	{
		return &List[ c->Next ];
	}

	bool isValid()
	{
		int p = FirstPossible;
		double s = List[p].Value;

		int c = 0;
		int j;

		for (j = 0; j < numPossible; j++)
		{
			if (List[p].Value >	s)
				return false;
			s = List[p].Value;
			
			if (!List[p].Possible || !List[p].Valid || List[p].Tabu != 0)
				return false;

			p = List[p].Next;
			
			c++;
		}

		if (p != LastPossible)
			return false;

		do 
		{
			
			if (List[p].Possible || ( List[p].Valid && List[p].Tabu == 0) )
				return false;

			p = List[p].Next;

			c++;

		} while ( p != FirstPossible);

		if ( c != numTotal )
			return false;

		return true;
	}

};


class GreedySearch;

#define BLACK 2
#define WHITE 0
#define GREY 1

class State
{
#ifdef GRAPH_SPACE
public:
	BOOL *Graph;

	double	***SetValuesPtr;//give the value of each set 
	double	***LSetValuesPtr;//give the value of each set 
	ADTree  *TreeTrain;
	ADTree  *TreeTest;
	Statistics *StatsTrain;
	Statistics *StatsTest;
	int		*AllDomains;

	int		maxParents;
	int		numVariables;
	int		tabuSize;

	double	*famScores;
	ListOp	*deltaScores;
	Operator	*tabuList;
	ListParent *Parents;
	int		*Marked;

	int		*SetReachedCountPtr;
	BOOL	***SetReachedPtr;

	int		numCandidates;
	int		**Candidates;
	int		**CandidatesNb;

	//TEMP
	int		*parent;
	int		*domains;

	State(int n, int l, GreedySearch *search);

	~State()
	{
		delete[] Marked;
		delete[] Graph;
		delete[] famScores;
		delete deltaScores;
		delete[] tabuList;
		delete[] Parents;
		delete[] parent;
		delete[] domains;
	}

	void CopyStateFrom(State *s)
	{
		ASSERT(numVariables==s->numVariables);
		int n;

		for ( n=0; n < numVariables*numVariables ; n++)
			Graph[n] = s->Graph[n];
	
		for ( n=0; n < numVariables ; n++)
		{
			famScores[n] = s->famScores[n];
			Parents[n].CopyFrom( &s->Parents[n] );
		}

		deltaScores->CopyAllExceptTabuFrom(s->deltaScores);
	}

        void Init(GreedySearch * search);

	void GetGraph(BOOL * g)
	{
		memcpy(g,Graph, sizeof(BOOL)*numVariables*numVariables);
	}	
	
	double GetScore(int maxParents)
	{
		double score = 0;

		for (int n = 0; n < numVariables ; n++)
			score += famScores[n];

		return score;
	}
	
	double GetFamScore(int n)
	{
		double score = 0;
		int np;
		int j;
		
		ElemParent *e;
		Counter count1;
		Counter count2;
		int p;
		
		np = Parents[n].GetNumParent();

		e = Parents[n].GetFirstParent();
		for ( j = 0; j < np; j++ )
		{
			//p = e->val;
			p = Candidates[n][e->val];
			domains[j] = AllDomains[p];
			parent[j] = p;
			
			e = Parents[n].GetNextParent(e);
		}

		int setParentPtr = SetToPtr(numVariables, np, parent);

		if ( !(*SetReachedPtr)[np][setParentPtr] )
		{

			(*SetReachedCountPtr)++;
			(*SetReachedPtr)[np][setParentPtr] = TRUE;
			count1.InitCounter(np, parent, domains, StatsTrain, VIRTUAL_COUNT, TreeTrain);
			count2.InitCounter(np, parent, domains, StatsTest, VIRTUAL_COUNT, TreeTest);
			(*SetValuesPtr)[np][setParentPtr] = count1.GetBayesianValue();
			(*LSetValuesPtr)[np][setParentPtr] = count2.GetLikelihoodValue();
		}
	
		score = -(*SetValuesPtr)[np][setParentPtr];
	
		int	decal = 0;
		e = Parents[n].GetFirstParent();
		for ( j = 0; j < np; j++ )
		{
			//p = e->val;
			p = Candidates[n][e->val];
			if (p>n && decal == 0)
			{
				domains[j] = AllDomains[n];
				parent[j] = n;
				decal = 1;
			}
			domains[ j + decal ] = AllDomains[p];
			parent[ j + decal ] = p;
			
			e = Parents[n].GetNextParent(e);
		}
		
		if (decal == 0)
		{
			domains[np] = AllDomains[n];
			parent[np] = n;
		}

		int setFamPtr = SetToPtr(numVariables, np+1, parent);
	
		if ( !(*SetReachedPtr)[np+1][setFamPtr] )
		{

			(*SetReachedCountPtr)++;
			(*SetReachedPtr)[np+1][setFamPtr] = TRUE;
			
			count1.InitCounter(np+1, parent, domains, StatsTrain, VIRTUAL_COUNT, TreeTrain);
			count2.InitCounter(np+1, parent, domains, StatsTest, VIRTUAL_COUNT, TreeTest);
	
			(*SetValuesPtr)[np+1][setFamPtr] = count1.GetBayesianValue();
			(*LSetValuesPtr)[np+1][setFamPtr] = count2.GetLikelihoodValue();

		}

		score += (*SetValuesPtr)[np+1][setFamPtr];

		return score;
	}

	double	GetLikelihood()
	{
		double value = 0;
		for (int i= 0; i < numVariables; i++)
			value += GetFamLikelihood(i);

		return value;
	}

	double GetFamLikelihood(int n)
	{
		double score = 0;
		int np;
		int j;
		
		ElemParent *e;
		Counter count;
		int p;
		
		np = Parents[n].GetNumParent();

		e = Parents[n].GetFirstParent();
		for ( j = 0; j < np; j++ )
		{
			//p = e->val;
			p = Candidates[n][e->val];
			domains[j] = AllDomains[p];
			parent[j] = p;
			
			e = Parents[n].GetNextParent(e);
		}

		int setParentPtr = SetToPtr(numVariables, np, parent);

		assert ( (*SetReachedPtr)[np][setParentPtr] );
		/*{

			(*SetReachedCountPtr)++;
			(*SetReachedPtr)[np][setParentPtr] = TRUE;
			count.InitCounter(np, parent, domains, Stats, VIRTUAL_COUNT, Tree);
			(*SetValuesPtr)[np][setParentPtr] = count.GetBayesianValue();
			(*LSetValuesPtr)[np][setParentPtr] = count.GetLikelihoodValue();
		}*/
	
		score = -(*LSetValuesPtr)[np][setParentPtr];
	
		int	decal = 0;
		e = Parents[n].GetFirstParent();
		for ( j = 0; j < np; j++ )
		{
			//p = e->val;
			p = Candidates[n][e->val];
			if (p>n && decal == 0)
			{
				domains[j] = AllDomains[n];
				parent[j] = n;
				decal = 1;
			}
			domains[ j + decal ] = AllDomains[p];
			parent[ j + decal ] = p;
			
			e = Parents[n].GetNextParent(e);
		}
		
		if (decal == 0)
		{
			domains[np] = AllDomains[n];
			parent[np] = n;
		}

		int setFamPtr = SetToPtr(numVariables, np+1, parent);
	
		assert( (*SetReachedPtr)[np+1][setFamPtr] );
		/*{

			(*SetReachedCountPtr)++;
			(*SetReachedPtr)[np+1][setFamPtr] = TRUE;
			
			count.InitCounter(np+1, parent, domains, Stats, VIRTUAL_COUNT, Tree);
	
			(*SetValuesPtr)[np+1][setFamPtr] = count.GetBayesianValue();
			(*LSetValuesPtr)[np+1][setFamPtr] = count.GetLikelihoodValue();

		}*/

		score += (*LSetValuesPtr)[np+1][setFamPtr];

		return score;
	}

	bool doOperator(Operator o, GreedySearch * search)
	{

		switch (o.type)
		{
		case ADD_EDGE:
			ASSERT(Graph[o.from * numVariables + o.to] == FALSE);
			Graph[o.from * numVariables + o.to] = TRUE;
			//Parents[o.to].Push(o.from);
			Parents[o.to].Push(o.cand);

			if (!CheckForAcyclicity())
			{
				Graph[o.from * numVariables + o.to] = FALSE;
				//Parents[o.to].Pop(o.from);
				Parents[o.to].Pop(o.cand);
				return false;
			}
		
			famScores[o.to] = GetFamScore(o.to);
			UpdateDeltaScore(o.to);
			break;
		case DELETE_EDGE:
			ASSERT(Graph[o.from * numVariables + o.to] == TRUE);
			Graph[o.from * numVariables + o.to] = FALSE;
			//Parents[o.to].Pop(o.from);
			Parents[o.to].Pop(o.cand);
			famScores[o.to] = GetFamScore(o.to);
			UpdateDeltaScore(o.to);
			break;
		case REVERSE_EDGE:
			ASSERT(Graph[o.from * numVariables + o.to] == TRUE);
			ASSERT(Graph[o.to * numVariables + o.from] == FALSE);
			ASSERT(o.cand != UNDEF);
			ASSERT(o.icand != UNDEF);

			Graph[o.from * numVariables + o.to] = FALSE;
			Graph[o.to * numVariables + o.from] = TRUE;
			Parents[o.from].Push(o.icand);
			Parents[o.to].Pop(o.cand);
			//Parents[o.from].Push(o.to);
			//Parents[o.to].Pop(o.from);
			if (!CheckForAcyclicity())
			{
				Graph[o.from * numVariables + o.to] = TRUE;
				Graph[o.to * numVariables + o.from] = FALSE;
				Parents[o.from].Pop(o.icand);
				Parents[o.to].Push(o.cand);
				//Parents[o.from].Pop(o.to);
				//Parents[o.to].Push(o.from);
				return false;
			}
			famScores[o.to] = GetFamScore(o.to);
			famScores[o.from] = GetFamScore(o.from);
			UpdateDeltaScore(o.to);
			UpdateDeltaScore(o.from);
			break;
		}

		return true;
	}

	//try to see is n can be put in a topological order (where parents are all nodes marked black
	//amd nodes marked grey can not be put now)
	bool CyclicVisit(int n)
	{
		if (Marked[n] == GREY)
			return false;

		if (Marked[n] == BLACK)
			return true;

		Marked[n] = GREY;

		ElemParent *e = Parents[n].GetFirstParent();
		int np = Parents[n].GetNumParent();

		for (int p =0; p < np; p++)
		{

			//if (!CyclicVisit(e->val))
			if (!CyclicVisit(Candidates[n][e->val]))
				return false;

			e = Parents[n].GetNextParent(e);
		}

		Marked[n] = BLACK;
		return true;
	}

	bool CheckForAcyclicity()
	{
		int n;
		for (n = 0 ; n < numVariables; n++)
		{
			Marked[n] = WHITE;
		}

		for (n = 0 ; n < numVariables; n++)
		{
			if (!CyclicVisit(n))
				return false;
		}

		return true;
	}

	void UpdateDeltaScore(int n)
	{

		int n2,k,k2;
		//for (n2 = 0; n2 < numVariables; n2++)
		for (k = 0; k < numCandidates; k++)
		{
			n2 = Candidates[n][k];
			k2 = CandidatesNb[n2][n];

			if (Graph[n2 * numVariables + n])
			{
				//modify delete score
				Graph[n2 * numVariables + n] = FALSE;
				Parents[n].Pop(k);
				//Parents[n].Pop(n2);

				deltaScores->SetScore( 
					DELETE_EDGE* numVariables * numCandidates + n * numCandidates + k ,
					//DELETE_EDGE* numVariables * numVariables + n * numVariables + n2 ,
					GetFamScore(n) - famScores[n]);
				
				if (
					k2 !=UNDEF && 
					deltaScores->isValid( ADD_EDGE* numVariables * numCandidates + n2 * numCandidates + k2 )
				//if (deltaScores->isValid( ADD_EDGE* numVariables * numCandidates + n2 * numVariables + n )
					 )
					deltaScores->SetScore(
						REVERSE_EDGE* numVariables * numCandidates + n * numCandidates + k ,
						//REVERSE_EDGE* numVariables * numVariables + n * numVariables + n2 ,
						GetFamScore(n) - famScores[n] + 
						//deltaScores->GetScore( ADD_EDGE* numVariables * numVariables + n2 * numVariables + n ) );
						deltaScores->GetScore( ADD_EDGE* numVariables * numCandidates + n2 * numCandidates + k2 ) );
			
				Graph[n2 * numVariables + n] = TRUE;
				Parents[n].Push(k);
				//Parents[n].Push(n2);
			}
			else
			{
				deltaScores->SetInvalid( DELETE_EDGE* numVariables * numCandidates + n * numCandidates + k );
				deltaScores->SetInvalid( REVERSE_EDGE* numVariables * numCandidates + n * numCandidates + k );
				//deltaScores->SetInvalid( DELETE_EDGE* numVariables * numVariables + n * numVariables + n2 );
				//deltaScores->SetInvalid( REVERSE_EDGE* numVariables * numVariables + n * numVariables + n2 );
			}	
		}

		//Too many parents ?
		if (Parents[n].GetNumParent() != maxParents-1)
		{
			//we can still add parents
			for (k = 0; k < numCandidates; k++)
			{
				n2 = Candidates[n][k];
				k2 = CandidatesNb[n2][n];
			//for (n2 = 0; n2 < numVariables; n2++)
			//{
				if (!Graph[n2 * numVariables + n])// && n != n2)
				{
					Graph[n2 * numVariables + n] = TRUE;
					//Parents[n].Push(n2);
					Parents[n].Push(k);

					deltaScores->SetScore( ADD_EDGE* numCandidates * numVariables + n * numCandidates + k  ,GetFamScore(n) - famScores[n]);
					//deltaScores->SetScore( ADD_EDGE* numVariables * numVariables + n * numVariables + n2  ,GetFamScore(n) - famScores[n]);
					
					if (k2 != UNDEF && deltaScores->isValid( DELETE_EDGE* numVariables * numCandidates + n2 * numCandidates + k2 ))
						deltaScores->SetScore( 
							REVERSE_EDGE* numVariables * numCandidates + n2 * numCandidates + k2 ,
							GetFamScore(n) - famScores[n] + 
							deltaScores->GetScore( DELETE_EDGE* numVariables * numCandidates + n2 * numCandidates + k2 ) );

					//if (deltaScores->isValid( DELETE_EDGE* numVariables * numVariables + n2 * numVariables + n ))
					//	deltaScores->SetScore( 
					///		REVERSE_EDGE* numVariables * numVariables + n2 * numVariables + n ,
					//		GetFamScore(n) - famScores[n] + 
					//		deltaScores->GetScore( DELETE_EDGE* numVariables * numVariables + n2 * numVariables + n ) );

					Graph[n2 * numVariables + n] = FALSE;
					//Parents[n].Pop(n2);
					Parents[n].Pop(k);
				}
				else
				{
					deltaScores->SetInvalid( ADD_EDGE* numVariables * numCandidates + n * numCandidates + k  );
					deltaScores->SetInvalid( REVERSE_EDGE* numVariables * numCandidates + n2 * numCandidates + k2   );
					//deltaScores->SetInvalid( ADD_EDGE* numVariables * numVariables + n * numVariables + n2  );
					//deltaScores->SetInvalid( REVERSE_EDGE* numVariables * numVariables + n2 * numVariables + n  );
				}
			}
		}
		else
		{
			for (k = 0; k < numCandidates; k++)
			{
				n2 = Candidates[n][k];
				k2 = CandidatesNb[n2][n];
		//	for (n2 = 0; n2 < numVariables; n2++)
		//	{
					deltaScores->SetInvalid( ADD_EDGE* numVariables * numCandidates + n * numCandidates + k  );
					deltaScores->SetInvalid( REVERSE_EDGE* numVariables * numCandidates + n2 * numCandidates + k2   );
					//deltaScores->SetInvalid( ADD_EDGE* numVariables * numVariables + n * numVariables + n2  );
					//deltaScores->SetInvalid( REVERSE_EDGE* numVariables * numVariables + n2 * numVariables + n  )
			}

		}
	}

	Operator Sample()
	{
		Operator o;
		o.type = (int) (RANDOM*NB_TYPE_OPERATOR);
		o.cand = (int) (RANDOM*numCandidates);
		o.to   = (int) (RANDOM*numVariables);
		o.from = Candidates[o.to][o.cand];
		o.icand = CandidatesNb[o.from][o.to];
		return o;
	}
	
	Operator Next(Operator o)
	{
		ASSERT(o.type != UNDEF);
		if (o.cand != numCandidates-1)
			o.cand++;
		else
		{
			o.cand = 0;
			if (o.to != numVariables-1)
				o.to++;
			else
			{
				o.to = 0;
				if (o.type != NB_TYPE_OPERATOR-1)
					o.type++;
				else
					o.type = 0;
			}
		}
		o.from = Candidates[o.to][o.cand];
		o.icand = CandidatesNb[o.from][o.to];
		/*if (o.from != numVariables-1)
			o.from++;
		else
		{
			o.from = 0;
			if (o.to != numVariables-1)
				o.to++;
			else
			{
				o.to = 0;
				if (o.type != NB_TYPE_OPERATOR-1)
					o.type++;
				else
					o.type = 0;
			}
		}*/
		return o;
	}

	int GetNumOp(Operator o)
	{
		return o.type*numVariables*numCandidates + o.to * numCandidates + o.cand;
		//return o.type*numVariables*numVariables + o.to * numVariables + o.from;
	}

	int GetNumberOp()
	{
		return 3*numVariables*numCandidates;
		//return 3*numVariables*numVariables;
	}

	Operator GetFirstOp()
	{
		Operator o;
		o.type = 0;
		//o.from = 0;
		o.cand = 0;
		o.to = 0;
		o.from = Candidates[o.to][o.cand];
		o.icand = CandidatesNb[o.from][o.to];
		return o;
	}

	//test everything before doing a move (for random restart)
	BOOL isLegalOp(Operator o, Operator *tabu, int t, int l,  int maxParents)
	{

		BOOL res;

		ASSERT(o.from != o.to);
//		if (o.from == o.to)
//			return FALSE;

		switch (o.type)
		{
		case ADD_EDGE:
			//Already present
			if (Graph[o.from * numVariables + o.to])
				return FALSE;
			//Too many parents
			if (Parents[o.to].GetNumParent() == maxParents-1)
				return FALSE;
			
			Graph[o.from * numVariables + o.to] = TRUE;
			//Parents[o.to].Push(o.from);
			Parents[o.to].Push(o.cand);
		
			res = CheckForAcyclicity();

			Graph[o.from * numVariables + o.to] = FALSE;
			//Parents[o.to].Pop(o.from);
			Parents[o.to].Pop(o.cand);
		
			return res;
		case DELETE_EDGE:
			if (!Graph[o.from * numVariables + o.to])
				return FALSE;
			return TRUE;
		case REVERSE_EDGE:
			if (!Graph[o.from * numVariables + o.to])
				return FALSE;
			//Too many parents
			if (Parents[o.from].GetNumParent() == maxParents-1)
				return FALSE;
			
			if ( o.icand == UNDEF)
				return FALSE;

			//Connexity test
			Graph[o.from * numVariables + o.to] = FALSE;
			Graph[o.to * numVariables + o.from] = TRUE;
			//Parents[o.from].Push(o.to);
			//Parents[o.to].Pop(o.from);

			Parents[o.from].Push(o.icand);
			Parents[o.to].Pop(o.cand);
			res = CheckForAcyclicity();

			Graph[o.from * numVariables + o.to] = TRUE;
			Graph[o.to * numVariables + o.from] = FALSE;
			//Parents[o.from].Pop(o.to);
			//Parents[o.to].Push(o.from);
			Parents[o.from].Pop(o.icand);
			Parents[o.to].Push(o.cand);
			
			return res;
		}

		return FALSE;
	}	

	void		IncTabuReverse(Operator o, int i)
	{
		if (o.type == UNDEF)
			return;

		deltaScores->IncTabu(GetNumOp(o), i);

		int t;

		switch (o.type)
		{
		case ADD_EDGE:
				o.type = DELETE_EDGE;
				deltaScores->IncTabu(GetNumOp(o), i);
				break;
		case DELETE_EDGE:
				o.type = ADD_EDGE;
				deltaScores->IncTabu(GetNumOp(o), i);
				break;
		case REVERSE_EDGE:
				t = o.to;
				o.to = o.from;
				o.cand = o.icand;
				//o.from = t;
				o.type = ADD_EDGE;
				deltaScores->IncTabu(GetNumOp(o), i);
				o.type = REVERSE_EDGE;
				deltaScores->IncTabu(GetNumOp(o), i);
				break;
		}

	}

	void	PutTabu(Operator o, int t)
	{
		IncTabuReverse(o,1);
		IncTabuReverse(tabuList[t], -1);
		tabuList[t] = o;
	}

	void	SetInvalid(Operator o)
	{
		deltaScores->SetInvalid( GetNumOp(o) );
	}

	Chain*	GetBestOperator()
	{
		return deltaScores->GetFirst();
	}

	Chain*  GetNextOperator(Chain *c)
	{
		return deltaScores->GetNext(c);
	}

#endif
	
#ifdef ORDER_SPACE
	

public:
	int *Rank;
	int *Ordering;
	int *BestFamilyPtr;
	Family ***FamilyList;
	int **ValidityCount;
	int *FamilyCount;
	double *deltaScores;

	int **Candidates;
	int numCandidates;

	int numVariables;
	//BOOL	*tabuGraph;

	State(int n, GreedySearch *search);

	~State()
	{
		for (int v=0; v < numVariables; v++)
		{
			delete ValidityCount[v];
			//delete[] FamilyList[v];
		}

		delete[] FamilyCount;
		delete[] deltaScores;
		delete[] ValidityCount;
		delete[] Rank;
		delete[] Ordering;
		delete[] BestFamilyPtr;
	}

	void CopyStateFrom(State *s)
	{
		int n;

		ASSERT(numVariables==s->numVariables);
		FamilyList = s->FamilyList;

		for (n=0; n < numVariables ; n++)
		{
			Rank[n] = s->Rank[n];
			Ordering[n] = s->Ordering[n];
			BestFamilyPtr[n] = s->BestFamilyPtr[n];
			//memcpy(FamilyList[n], s->FamilyList[n], sizeof(Family)*FamilyCount[n] );
			memcpy(ValidityCount[n], s->ValidityCount[n], sizeof(int)*FamilyCount[n] );
		}

		for (n=0; n< numVariables-1; n++)
			deltaScores[n] = s->deltaScores[n];
	}

	void Init(GreedySearch *search);
	
	void UpdateDeltaScore( int n, int** InFamilyCount, int ***InFamilyPtr);

	double	GetDeltaScore( Operator o);
	
	int GetNumOp( Operator o);
	
	Operator GetFirstOp();

	double GetPotentialDeltaDown( int n, int n2, int** InFamilyCount, int ***InFamilyPtr);
	
	double GetPotentialDeltaUp( int n, int n2, int** InFamilyCount, int ***InFamilyPtr);

	void GetGraph(BOOL * g)
	{
		int n,n2,p,v;

                // Clear all edges from graph.
		for (n = 0; n < numVariables ; n++)		
			for ( n2 = 0; n2 < numVariables; n2++)
				g[n2 * numVariables + n] = FALSE;
			
		Family *f;
		int		*Set;

		for (n = 0; n < numVariables ; n++)
		{
			/*f = NULL;

			for (p = 0; p < FamilyCount[n]; p++)
			{
				f = &FamilyList[n][p];
				for ( n2 = 0; n2 < f->numParents; n2++)
				{
					if (Set[n2] < n)
					{
						if (Set[n2] != f->Parents[n2])
							printf("ERROR");
					}
					else
					{
						if (Set[n2]+1 != f->Parents[n2])
							printf("ERROR");
					}
					
					if (Rank[f->Parents[n2]] > Rank[n])
						break;
				}
				//delete Set;
				if (n2 == f->numParents)
					break;
			}
			ASSERT(p != FamilyCount[n]);
			ASSERT(p == BestFamilyPtr[n]);
			*/

			p = BestFamilyPtr[n];
			ASSERT(FamilyList != NULL);

			f = &((*FamilyList)[n][p]);
			Set = new int[f->numParents];
			PtrToSet(f->ptr, numCandidates, f->numParents, Set);

			for ( n2 = 0; n2 < f->numParents; n2++)
			{
				v = Candidates[n][Set[n2]];
				//ASSERT( v == f->Parents[n2]);  // v is a parent of n
				g[v * numVariables + n] = TRUE;
			}
			delete Set;
		}
	}

	double GetScore( int maxParents)
	{
		double score = 0;
		//Family *f;

		for (int n = 0; n < numVariables ; n++)
		{
			/*f = NULL;

			for (int p = 0; p < FamilyCount[n]; p++)
			{
				f = FamilyList[n][p];
				for (int n2 = 0; n2 < f->numParents; n2++)
					if (Rank[f->Parents[n2]] > Rank[n])
						break;
				if (n2 == f->numParents)
					break;
			}
			
			ASSERT(p != FamilyCount[n]);
			ASSERT( p == BestFamilyPtr[n]);*/
			
			score += (*FamilyList)[n][ BestFamilyPtr[n] ].score;//f->score;
		}

		return score;
	}

	double GetLikelihood()
	{
		double score = 0;
		//Family *f;

		for (int n = 0; n < numVariables ; n++)
		{
			/*f = NULL;

			for (int p = 0; p < FamilyCount[n]; p++)
			{
				f = FamilyList[n][p];
				for (int n2 = 0; n2 < f->numParents; n2++)
					if (Rank[f->Parents[n2]] > Rank[n])
						break;
				if (n2 == f->numParents)
					break;
			}
			
			ASSERT(p != FamilyCount[n]);
			ASSERT( p == BestFamilyPtr[n]);*/
			
			score += (*FamilyList)[n][ BestFamilyPtr[n] ].likelihood;//f->score;
		}

		return score;
	}
	/*
	int GetFamPtr(int n)
	{
		int p,n2;

		//double score = 0;
		Family *f;

		f = NULL;

		for ( p = 0; p < FamilyCount[n]; p++)
		{
			f = &FamilyList[n][p];
			for ( n2 = 0; n2 < f->numParents; n2++)
				if (Rank[f->Parents[n2]] > Rank[n])
				{
					ASSERT(ValidityCount[n][p] == 0);
					break;
				}
				else
					ASSERT(ValidityCount[n][p] != 0);

			if (n2 == f->numParents)
				break;
		}
		ASSERT(p != FamilyCount[n]);
		
		
		return p;		
	}*/
	void UpdateFamPtrUp(int n, int n2, int** InFamilyCount, int ***InFamilyPtr);

	void UpdateFamPtrDown(int n, int n2, int** InFamilyCount, int ***InFamilyPtr);

	
/*#ifdef ORDER_SPACE2

	void doOperator(Operator o, Family ***FamilyTree, int ** InFamilyCount, int *** InFamilyPtr, int maxParents)
	{
 		ASSERT(o.type == REVERSE_NODE);
		ASSERT(o.from == Rank[o.var]);

		int v;


		if (o.from < o.to)
		{
			for ( v = o.from ; v < o.to ; v++)
			{
				Ordering[v] = Ordering[v+1];
				Rank[Ordering[v]]--; 
			}
		}
		else
		{
			for ( v = o.from; v > o.to  ; v--)
			{
				Ordering[v] = Ordering[v-1];
				Rank[Ordering[v]]++; 
			}
		}
		
		Ordering[o.to] = o.var;
		Rank[o.var] = o.to;

		int min = MIN(o.from, o.to);
		int max = MAX(o.from,o.to);
		for ( v = min;  v <= max  ; v++)
			BestFamilyPtr[Ordering[v]] = GetFamPtr(Ordering[v], FamScores, maxParents); 
		
		//DEBUG
		//GetScore(FamScores, FamilyCount, FamilyList, maxParents);
	}

	void undoOperator(Operator o, Family ***FamilyTree, int ** InFamilyCount, int *** InFamilyPtr, int maxParents)
	{
		ASSERT(o.type == REVERSE_NODE);
		int v;
		if (o.from < o.to)
		{
			for ( v = o.to ; v >= o.from + 1 ; v--)
			{
				Ordering[v] = Ordering[v-1];
				Rank[Ordering[v]]++; 
			}
		}
		else
		{
			for ( v = o.to ; v <= o.from - 1 ; v++)
			{
				Ordering[v] = Ordering[v+1];
				Rank[Ordering[v]]--; 
			}
		}
		Ordering[o.from] = o.var;
		Rank[o.var] = o.from;

		int min = MIN(o.from, o.to);
		int max = MAX(o.from,o.to);
		for ( v = min;  v <= max  ; v++)
			BestFamilyPtr[Ordering[v]] = GetFamPtr(Ordering[v], FamScores, maxParents); 
	}

	Operator Sample()
	{
		Operator o;
		o.type = (int) (RANDOM*NB_TYPE_OPERATOR);
		o.var = (int) (RANDOM*numVariables);
		o.from = Rank[o.var];
		o.to   = (int) (RANDOM*numVariables);
		return o;
	}

	Operator Next(Operator o)
	{
		ASSERT(o.type != UNDEF);
		if (o.var != numVariables-1)
			o.var++;
		else
		{
			o.var = 0;
			if (o.to != numVariables-1)
				o.to++;
			else
			{
				o.to = 0;
				if (o.type != NB_TYPE_OPERATOR-1)
					o.type++;
				else
					o.type = 0;
			}
		}
		o.from = Rank[o.var];

		return o;
	}


	BOOL isLegalOp(Operator o, Operator *tabu, int t, int l,  int maxParents)
	{
	
		ASSERT(o.from == Rank[o.var]);

		if (o.from == o.to)
			return FALSE;

		switch (o.type)
		{
		case REVERSE_NODE:
			//always legal
			break;
		default:
			ASSERT(FALSE);
		}

		//Not in tabu
		for (int i = 0 ; i < l; i++)
			if (
#ifdef NO_REVERSE
				o.isReverseOf(tabu[i]) ||
#endif
				o.isEqualTo(tabu[i]))
				return FALSE;
		return TRUE;

	}	
#endif*/

//#ifdef ORDER_SPACE3

	void doOperator(Operator o, GreedySearch * search);

	void undoOperator(Operator o, GreedySearch * search);


	Operator Sample()
	{
		Operator o;
		o.type = (int) (RANDOM*NB_TYPE_OPERATOR);
		o.pos = (int) (RANDOM*(numVariables-1));
		o.var1 = Ordering[o.pos];
		o.var2 = Ordering[o.pos+1];
		return o;
	}

	
	Operator Next(Operator o)
	{
		ASSERT(o.type != UNDEF);
		if (o.pos != numVariables-2)
			o.pos++;
		else
		{
			o.pos = 0;
		}
		o.var1 = Ordering[o.pos];
		o.var2 = Ordering[o.pos+1];
		return o;
	}



	BOOL isLegalOp(Operator o, Operator *tabu, int t, int l,  int maxParents)
	{
	
		ASSERT(o.pos == Rank[o.var1]);
		ASSERT(o.pos+1 == Rank[o.var2]);
		ASSERT(o.var1 != o.var2);
		ASSERT(o.type == REVERSE_NODE);

		//Not in tabu
		for (int i = 0 ; i < l; i++)
			if (o.isReverseOf(tabu[i]))
				return FALSE;
		return TRUE;

	}	
//#endif
#endif    // #ifdef ORDER_SPACE
	

};

class GreedySearch
{
	int		numVariables;
	double	currentTime;
	clock_t		StartTime;
	double	maxFoundScore;

public:
	Statistics		*StatsTrain;
	Statistics		*StatsTest;
	int	maxParents; //it is a strict max == depth of the tree
	int				SetReachedCount;
	int		numCandidates;
	int		**Candidates;
	int		**CandidatesNb;

#ifdef GRAPH_SPACE
	BOOL			**SetReached;
	double			**SetValues;
	double			**LSetValues;
	ADTree *TreeTest;
	ADTree *TreeTrain;
#endif

#ifdef ORDER_SPACE
	//int				***FamScoreLinks;//FamScoreLinks[d][i][k] : gives the ptr to the next FamScoreNode (independent of n) ! 
	Family			**FamilyList;//sorted list of possible Family for each nodes
	int				*FamilyCount;//count of family in FamilyList
	int				**InFamilyCount;
	int				***InFamilyPtr;//pointers in FamilyList of Families that of i that have parent j
#endif


public:

	void	UpdateTime()
	{		
		currentTime += ((double)(clock()-StartTime))/CLOCKS_PER_SEC;
		StartTime = clock();
	}
	
	~GreedySearch()
	{

		int v;

#ifdef GRAPH_SPACE
		int d;


		for ( d = 0 ; d < maxParents + 1; d++)
		{

			delete[]	SetReached[d] ;
			delete[]	LSetValues[d];//FamilyTree[v][d] ;
			delete[]	SetValues[d];//FamilyTree[v][d] ;
		}
		delete	LSetValues;//		delete	FamilyTree;
		delete	SetValues;//		delete	FamilyTree;
		delete  SetReached;
		delete TreeTrain;
		delete TreeTest;

#endif


		for ( v = 0; v < numVariables; v++)
		{
		
#ifdef ORDER_SPACE	
			
			for (int v2 = 0; v2 < numVariables; v2++)
			{
				delete[]	InFamilyPtr[v][v2];
			}
			delete	FamilyList[v];
			delete  InFamilyCount[v];
			delete  InFamilyPtr[v];
#endif

			delete Candidates[v];
			delete CandidatesNb[v];
	

		}

#ifdef ORDER_SPACE
		delete	FamilyList;
		delete	FamilyCount;
		delete  InFamilyCount;
		delete  InFamilyPtr;
#endif

		delete Candidates;
		delete CandidatesNb;

	}

	//Family Sentinel;

/*	GreedySearch(int nvar, int maxparents, Statistics* stats, int rmin)
	{
		Init(nvar,maxparents,stats,rmin,20);
	}*/

  GreedySearch(int nvar, int maxparents, Statistics* stats1, Statistics* stats2,
               int rmin, int numcandidates) {
    Init(nvar,maxparents,stats1, stats2,rmin,numcandidates);
  }

  void	Init(int nvar, int maxparents, Statistics* stats1, Statistics* stats2,
             int rmin, int numcandidates) {
    int	s;
    int k,d;
    
    StartTime = clock();
    currentTime = 0;
		
    //printf("Precomputing ADTree...\n");
    int start = clock();
			
    int dmax = 0;
    for (k = 0; k < nvar; k++)
      if ( stats1->GetVariableDomain(k) > dmax)
        dmax = stats1->GetVariableDomain(k);
    
    poolCount = new int[dmax * (maxparents+1)];
    poolRecords = new int*[dmax * (maxparents+1) ];
    for (k = 0; k < dmax * (maxparents+1); k++)
      poolRecords[k] = new int[stats1->GetnumSamples()];
    
    int v;
    
#ifdef ORDER_SPACE
    int		f,p,s2;
    ADTree *TreeTest;
    ADTree *TreeTrain;
    double			**LSetValues;//give the value of each set 
    double			**SetValues;//give the value of each set 
    BOOL			**SetReached; 
    FamilyList = NULL;
#endif
    
    int r;
    
    for ( r = 0; r < stats1->GetnumSamples(); r++)
      poolRecords[0][r] = r;
    
    poolCountPtr = 0;
    poolRecordsPtr = 1;
    
    TreeTrain = new ADTree(0 , poolRecords[0] , stats1->GetnumSamples(), stats1, maxparents, rmin);
    
    
    for (k = 0; k < dmax * (maxparents+1); k++) {
      delete[] poolRecords[k];
      poolRecords[k] = new int[stats2->GetnumSamples()];
    }

    for ( r = 0; r < stats2->GetnumSamples(); r++)
      poolRecords[0][r] = r;
    
    poolCountPtr = 0;
    poolRecordsPtr = 1;
    
    TreeTest = new ADTree(0 , poolRecords[0] , stats2->GetnumSamples(), stats2, maxparents, rmin);
    
    for (k = 0; k < dmax * (maxparents+1); k++)
      delete[]  poolRecords[k];
    
    delete[] poolRecords;
    delete[] poolCount;
    
    //printf( "%f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC );
    
    maxParents = maxparents;
    numVariables = nvar;
    numCandidates = (numcandidates==UNDEF || numcandidates > numVariables-1)?numVariables-1:numcandidates;
    
    start = clock();
    
    //Create SetValues
    int		setPtr;
    SetReachedCount = 0;
    
    SetValues = new double*[maxParents+1];
    LSetValues = new double*[maxParents+1];
    SetReached = new BOOL*[maxParents+1];
    for ( d = 0 ; d < maxParents+1 ; d++) {
      s = Combin[numVariables][d];
      LSetValues[d] = new double[s];
      SetValues[d] = new double[s];
      SetReached[d] = new BOOL[ s ];
      for ( setPtr = 0; setPtr < s ; setPtr++)
        SetReached[d][setPtr] = FALSE;
    }

    
    UpdateTime();
    
    StatsTrain = stats1;
    StatsTest = stats2;
    
    //printf("Computing Set Values...\n");
    start = clock();
    
    //#endif
    
    int*	Set;
    //int		setPtr;
    int*	Domains;
    Counter count1;
    Counter count2;
    
    /*Set = new int[maxParents];
      Domains = new int[maxParents];
      
      for (d = maxParents  ; d >= 0 ; d--)
      {
      s = Combin[numVariables][d];
      //SetValues[d] = new double[s];
      
      for ( k = 0; k < d; k++)
      Set[k] = k;
      
      SetReachedCount += s;
      for ( setPtr = 0; setPtr < s ; setPtr++)
      {
      
      //				if ((setPtr&0x1FFF) == 0)
      //					printf("%d/%d  (%d/%d)\n",setPtr,s,maxParents-d,maxParents);
      
      for ( k = 0; k < d ; k++)
      Domains[k] = stats->GetVariableDomain( Set[k] );
      
      count.InitCounter(d, Set, Domains, stats, VIRTUAL_COUNT, Tree);
      
      SetValues[d][setPtr] = count.GetBayesianValue();
      
      NextSet(numVariables , d , Set);
      }			
      
      
      }
      
      //#ifdef ORDER_SPACE
      //#endif
      delete[] Set;
      delete[] Domains;*/
    
    //printf( "%f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC );
    //start = clock();
    //printf("Computing Family scores...\n");
    
    int*	parentSet;
    int*	parentSet2;
    int*	childrenSet;
    
    int		parentPtr;
    int		childrenPtr;
    
    
    //Create Nodes and FamilyList
    //and Populate Nodes
    double	score;
    
    int	j,c,m1,m2;
    
#ifdef ORDER_SPACE
    Family fam;
    int		parentPtr2;
    int ptr,v2,vv;
    double maxscore, score2;
    Family			**FamilyTree;//FamScores[d][i] : gives the ith family of child n containing d parents
    double			**MaxScores;//FamScores[d][i] : gives the ith family of child n containing d parents
    
    FamilyTree = new Family*[maxParents];
    MaxScores = new double*[maxParents-1];
    FamilyList = new Family*[numVariables];
    FamilyCount = new int[numVariables];
    for ( d = 0 ; d < maxParents ; d++) {
      s = Combin[numVariables-1][d];
      FamilyTree[d] = new Family[ s ];
    }
    for ( d = 0 ; d < maxParents-1 ; d++) {
      s = Combin[numVariables-1][d];
      MaxScores[d] = new double[ s ];
    }
#endif
    
    /*#ifdef GRAPH_SPACE
      FamilyTree = new Family**[numVariables];
      #endif*/
    
    UpdateTime();
    
    parentSet = new int[maxParents];
    parentSet2 = new int[maxParents];
    Domains = new int[maxParents];
    
    childrenSet = new int[maxParents+1];
    
    
    Candidates = new int*[numVariables];
    CandidatesNb = new int*[numVariables];
    for ( v = 0; v < numVariables; v++) {
      Candidates[v] = new int[numCandidates];
      CandidatesNb[v] = new int[numVariables];
      for ( k = 0 ; k < numVariables; k++)
        CandidatesNb[v][k] = UNDEF;
    }
    
    if (numcandidates == UNDEF) {
      for ( v = 0; v < numVariables; v++) {
        for ( k = 0 ; k < numVariables - 1; k++) {
          Candidates[v][k] = (k>=v)?k+1:k;
          CandidatesNb[v][(k>=v)?k+1:k] = k;
        }
      }
    } else {
      Set = new int[maxParents];
      //COMPUTE SETS FOR ALL SETS OF SIZE 0, 1 and 2
      for (d = 2  ; d >= 0 ; d--) {
        s = Combin[numVariables][d];
        //SetValues[d] = new double[s];
        
        for ( k = 0; k < d; k++)
          Set[k] = k;
        
        SetReachedCount += s;
        for ( setPtr = 0; setPtr < s ; setPtr++) {
          
//				if ((setPtr&0x1FFF) == 0)
//					printf("%d/%d  (%d/%d)\n",setPtr,s,maxParents-d,maxParents);

          for ( k = 0; k < d ; k++)
            Domains[k] = stats1->GetVariableDomain( Set[k] );
          
          count1.InitCounter(d, Set, Domains, stats1, VIRTUAL_COUNT, TreeTrain);
          count2.InitCounter(d, Set, Domains, stats2, VIRTUAL_COUNT, TreeTest);
          
          SetValues[d][setPtr] = count1.GetBayesianValue();
          LSetValues[d][setPtr] = count2.GetLikelihoodValue(count1);
          SetReached[d][setPtr] = TRUE;
          
          NextSet(numVariables , d , Set);
        }			
      }
      
      double	*BestScores = new double[numCandidates];
      int		*BestCandidates;
      //BUILDING CANDIDATe sets
      for ( v = 0; v < numVariables; v++) {
          
        BestCandidates = Candidates[v];
        
        for ( k = 0 ; k < numCandidates; k++) {
          BestScores[k] = SCORE_MIN;
        }
        
        for ( k = 0 ; k < numVariables; k++) {
          
          if (k!=v) {
            //parentPtr = combin(n,1)-combin(n-1-k,1)-1=n-(n-1-k)-1=k
            //childrenPtr = combin(n,2)-combin(n-1-min(k,v),2)-combin(n-1-max(k,v),1)-1
            //			  = n(n-1)/2-(n-1-m)(n-2-m)/2-(n-1-M)-1
            //			  = n2/2 -n/2 -n(n-2-m)/2 +(n-2-m)/2+m(n-2-m)/2-n+1+M-1
            //			  = n2/2 -n/2 -n2/2 +n +nm/2 +n/2 -1 -m/2 +mn/2 -m -m2/2 -n +M
            //			  = nm - 3/2m -1 -m2/2 +M
            //			  = m(2n-3-m)/2-1+M
            m1 = (k>v)?v:k;
            m2 = (k>v)?k:v;
            parentPtr = k;
            childrenPtr = m1 * ( 2*numVariables - 3 - m1) / 2 -1 +m2;
            
            score = SetValues[2][ childrenPtr] - SetValues[1][ parentPtr ];
            
            for (c = numCandidates - 1; c >= 0 && score > BestScores[c]; c--) {
              if (c!=numCandidates - 1) {
                BestScores[c+1] = BestScores[c]; 
                BestCandidates[c+1] = BestCandidates[c];
              }
            }
            //the new parent should be put in position c+1
            if (c!=numCandidates-1) {
              BestCandidates[c+1] = k;
              BestScores[c+1] = score;
            }
          }
          
        }
	
        /*for ( k = 0 ; k < numCandidates; k++)
          {
          printf("%d %d %d[%f]\n", v , k , BestCandidates[k], BestScores[k]);
          }*/
	
	
        //We should sort the Candidates (for computinag set ptrs easily)
        for ( k = 0 ; k < numCandidates; k++) {
          for ( j = numCandidates - 1; j>k ; j--) {
            if (BestCandidates[j] < BestCandidates[j-1]) {
              c = BestCandidates[j];
              BestCandidates[j] = BestCandidates[j-1];
              BestCandidates[j-1] = c;
            }
          }
          CandidatesNb[v][ BestCandidates[k] ] = k;
          //	printf("%d %d %d\n", v , k , BestCandidates[k]);
        }
      }
      
      delete BestScores;
      delete Set;
      
    }
    
    UpdateTime();
    
#ifdef GRAPH_SPACE
    delete parentSet;
    delete parentSet2;
    delete Domains;
    delete childrenSet;
#endif
    
#ifdef ORDER_SPACE
    //FamScores = new FamScoreNode**[numVariables];
    for ( v = 0; v < numVariables; v++) {

      /*#ifdef GRAPH_SPACE
        FamilyTree[v] = new Family*[maxParents];
        #endif*/
      
      //#ifdef ORDER_SPACE
      s2 = 0;
      FamilyCount[v] = 0;
      //#endif
      
      for ( d = 0 ; d < maxParents ; d++) {
        s = Combin[numCandidates][d];
        //s = Combin[numVariables-1][d];
        s2 += s;
        
        /*#ifdef GRAPH_SPACE
          FamilyTree[v][d] = new Family[s];
          #endif*/
        
      }
      
      //#ifdef ORDER_SPACE
      FamilyList[v] = new Family[s2];
      //#endif
      for (d = 0 ; d < maxparents ; d++) {
        s = Combin[numCandidates][d];
        //s = Combin[numVariables-1][d];
	
        for ( k = 0; k < d ; k++)
          parentSet[k] = k;
        
	
        //We iterate trough every family
        for (  f = 0; f < s; f ++) {
          UpdateTime();
          
          //DEBUG
          //if ((f&0x1FFF) == 0)
          //	printf("%d/%d  (%d/%d)\n",f,s,maxParents-d,maxParents);
          
          p = -1;
          
          for ( k = 0; k < d ; k++) {
            v2 = Candidates[v][parentSet[k]];
            if ( v2 < v) {
              childrenSet[k] = parentSet2[k] = v2;
              p = k;
            } else {
              childrenSet[k+1] = parentSet2[k] = v2;
            }
            
            /*if ( parentSet[k] < v)
              {
              childrenSet[k] = parentSet2[k] = parentSet[k];
              p = k;
              }
              else
              {
              childrenSet[k+1] = parentSet2[k] = parentSet[k]+1;
              }*/
          }
          
          childrenSet[p+1] = v; 
          
          childrenPtr = SetToPtr(numVariables  , d+1 , childrenSet);
          parentPtr = SetToPtr(numCandidates , d, parentSet);
          parentPtr2 = SetToPtr(numVariables , d, parentSet2);
          
          if (!SetReached[d+1][childrenPtr]) {
            
            for ( k = 0; k < d+1 ; k++)
              Domains[k] = stats1->GetVariableDomain( childrenSet[k] );
            
            count1.InitCounter(d+1, childrenSet, Domains, stats1, VIRTUAL_COUNT, TreeTrain);
            count2.InitCounter(d+1, childrenSet, Domains, stats2, VIRTUAL_COUNT, TreeTest);
            
            LSetValues[d+1][childrenPtr] = count2.GetLikelihoodValue(count1);
            SetValues[d+1][childrenPtr] = count1.GetBayesianValue();
            //LSetValues[d+1][childrenPtr] = count2.GetLikelihoodValue(count1);
            
            SetReached[d+1][childrenPtr] = TRUE;
            
            SetReachedCount++;
          }
          
          if (!SetReached[d][parentPtr2]) {
            
            for ( k = 0; k < d ; k++)
              Domains[k] = stats1->GetVariableDomain( parentSet2[k] );
            
            count1.InitCounter(d, parentSet2, Domains, stats1, VIRTUAL_COUNT, TreeTrain);
            count2.InitCounter(d, parentSet2, Domains, stats2, VIRTUAL_COUNT, TreeTest);
            
            LSetValues[d][parentPtr2] = count2.GetLikelihoodValue(count1);
            SetValues[d][parentPtr2] = count1.GetBayesianValue();
            //LSetValues[d][parentPtr2] = count2.GetLikelihoodValue(count1);
            
            SetReached[d][parentPtr2] = TRUE;
            
            SetReachedCount++;
          }
          
          
          score = SetValues[d+1][childrenPtr] - SetValues[d][parentPtr2];
          
          fam.score = score;
          fam.likelihood = LSetValues[d+1][childrenPtr] - LSetValues[d][parentPtr2];
          //iterate through children to find the max score of max scores of children
          //#ifdef ORDER_SPACE
          fam.numParents = d;
          fam.ptr = parentPtr;
          maxscore = SCORE_MIN;
          
          if ( d != 0) {
            //we consider every possible family reduction
            //int p = 0;//we are trying to add an element before the pth parent of parentSet
            for (vv = 0; vv < d; vv++) {
              //we are deleting the vth variable from parentSet
              for (v2 = 0; v2 < d; v2++) {
                if ( v2 < vv)
                  parentSet2[v2] = parentSet[v2];
                else if (v2 > vv)
                  parentSet2[v2-1] = parentSet[v2];
              }
              //the kth variable is a possible augmentation
              
              
              ptr = SetToPtr( numCandidates, d - 1, parentSet2);//FamScoreLinks[ d ][ parentPtr ][ v ];
              score = MaxScores[ d-1 ][ ptr ];
              score2 = FamilyTree[ d-1 ][ ptr ].score;
              
              if (score > maxscore)
                maxscore = score;
              if (score2 > maxscore)
                maxscore = score2;
              
              
            }	
          }
          
          if (d!=maxParents-1)
            MaxScores[ d ][ parentPtr ] = maxscore;
          
          FamilyTree[ d ][ parentPtr ] = fam;
          
          if (fam.score > maxscore) {
              /*printf("%d : %d [",v, FamilyCount[v]);
                for ( v2 = 0; v2 < d; v2++)
                printf("%d,",Candidates[v][parentSet[v2]]);
                printf("]\n");*/
            FamilyList[ v ] [ FamilyCount[v]++ ] = fam;
            
          }
          //#endif
          
          /*#ifdef GRAPH_SPACE
            FamilyTree[ v ][ d ][ parentPtr ] = fam;
            #endif*/
          
          
          NextSet(numCandidates, d , parentSet);
          
        }
        
	
      }
      
      //#ifdef ORDER_SPACE
      
      FamilyList[ v ]  = (Family*) realloc(FamilyList[v], sizeof(Family) * FamilyCount[v]);
      //printf("%d	%d\n", v, FamilyCount[v]);
      
      //#endif
      
      
      
      //delete childrenDomains;
      
      
    }
    
    
    //#ifdef ORDER_SPACE
    delete TreeTrain;
    delete TreeTest;
    
    for ( d = 0 ; d < maxParents-1 ; d++)
      delete	MaxScores[d] ;
    delete MaxScores;	
    
    for ( d = 0 ; d < maxParents ; d++)
      delete	FamilyTree[d] ;
    delete FamilyTree;	
    
    for ( d = 0 ; d < maxParents+1; d++) {
      delete SetReached[d];
      delete SetValues[d];
      delete LSetValues[d];
    }
    delete[] SetReached;
    delete[] SetValues;
    delete[] LSetValues;
    //#endif
    
    delete parentSet;
    delete parentSet2;
    delete childrenSet;
    delete Domains;
    
    
    //printf( "%f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC );
    
    
    //#ifdef ORDER_SPACE
    start = clock();
    
    //printf("Sorting and scanning families...\n");
    
    //Sort Family list using 
    for (v = 0; v < numVariables; v++)
      quickSort(FamilyList[v],0,FamilyCount[v]-1);
    
    //printf("Cleaning families...\n");
    //Delete unnecessary families
    //for (v = 0; v < numVariables; v++)
    //	CleanFamilyList(v);
    
    
    //Populating InFamily and not in famlily
    int c1,c2,v3;
    Family * fa;
    InFamilyCount = new int*[numVariables];
    //	NotInFamilyCount = new int*[numVariables];
    //InFamily = new Family***[numVariables];
    InFamilyPtr = new int**[numVariables];
    //	NotInFamily = new Family***[numVariables];
    Set = new int[maxParents];
    for (v = 0; v < numVariables; v++) {
      InFamilyCount[v] = new int[numVariables];
      //		NotInFamilyCount[v] = new int[numVariables];
      //InFamily[v] = new Family**[numVariables];
      InFamilyPtr[v] = new int*[numVariables];
      //		NotInFamily[v] = new Family**[numVariables];
      for ( v2 = 0; v2 < numVariables; v2++) {
        
        //InFamily[v][v2] = new Family*[FamilyCount[v]];
        InFamilyCount[v][v2]=0;
        InFamilyPtr[v][v2] = new int[FamilyCount[v]];
      }
      for ( k = 0 ; k < numCandidates; k++) {
        v2 = Candidates[v][k];
	//			NotInFamily[v][v2] = new Family*[FamilyCount[v]];
        c1 = 0;
        c2 = 0;
        for ( p = 0; p < FamilyCount[v]; p++) {
          fa = &FamilyList[ v ][ p ];
          
          PtrToSet(fa->ptr, numCandidates, fa->numParents, Set);
          
          for ( j = 0; j < fa->numParents; j++) {
            v3 = Candidates[v][Set[j]];//(Set[j]<v)?Set[j]:Set[j]+1;
            //v3 = (Set[j]<v)?Set[j]:Set[j]+1;
            //ASSERT( v == f->Parents[n2]);
            //g[v * numVariables + n] = TRUE;
            if ( v3 == v2)
              break;
            //if ( fa->Parents[j] == v2)
            //	break;
          }
          if (j!=fa->numParents) {
            InFamilyPtr[v][v2][c1++] = p;
            //InFamily[v][v2][c1++] = fa;
          }
          //					NotInFamily[v][v2][c2++] = f;
          //				else
          //					InFamily[v][v2][c1++] = f;
          
        }
        InFamilyCount[v][v2] = c1;
	//			NotInFamilyCount[v][v2] = c2;
      }
    }
    
    delete Set;
    
    //printf( "%f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC );
    //BOOL * Graph = BOOL[numVariables*numVariables];
    //for (v = 0; v < numVariables; v++)
    //	for (int v2 = 0 ; v2 < numVariables; v2++)
    //		if (InFamilyCount[v][v2]!=0)
    //			printf("%d <- %d\n",v,v2);
#endif
  }
  
  
	//Delete unnecessary list of the Familylist
	/*void CleanFamilyList(int v)
	{
		Family *f1,*f2;
		int count = 0;

		int p,p2,e1,e2;

		for ( p = 0; p < FamilyCount[v]; p++)
		{
			f1 = &FamilyList[ v ][ p ];
			printf("Family %d of %d : %f (",p, v, f1->score);
			for (int e = 0; e < f1->numParents; e++)
				printf("%d ", f1->Parents[e]);
			printf(")");

			//if there is a family f2 of higher score included in f1
			//then discard f1
			for ( p2 = 0; p2 < count ; p2++)
			{
				
				f2 = FamilyList[ v ][ p2 ];

				//if f2 included in f1, discard f1
				for ( e1 = 0, e2 = 0; e2 < f2->numParents ; e2++)
				{
					while ( e1 < f1->numParents && f1->Parents[e1] < f2 ->Parents[e2])
						e1++;
					if ( e1 != f1->numParents )
					{
						if (f1->Parents[e1] == f2 ->Parents[e2])
							e1++;
						else 
							break;
					}
					else
						break;
				}
				if ( e2 == f2->numParents )//f2 included in f1
					break;
			}
			//if (p2 == count)//we keep f1
			if (f1->score > f1->maxscore)
			{
				//if (f1->score < f1->maxscore)
				//	printf("ERROR");
				//if (v==2 && count == 323)
				//	printf("???");

				FamilyList[ v ] [count++] = f1;
			}
			//else
			//{
			//	if (f1->score > f1->maxscore)
			//		printf("ERROR");
			//}
			//	printf(" x");
			//printf("\n");
		}

		printf("Family of %d : %d => %d\n",v, FamilyCount[v], count);

		//add sentinel
		FamilyList[ v ] [ count ] = &Sentinel;

		FamilyCount[v] = count;

	}*/

	/*BOOL isConsistent(int * Rank, int* initFam)
	{
		int i, j, k;

		for ( i=0; i < numVariables; i++)
		{
			if (Rank[i]!=UNDEF)
			{
				Family * f= FamilyList[i][ initFam[i] ];
				for ( j = 0; j < f->numParents; j++)
				{
					k = f->Parents[j];

					if (Rank[k]==UNDEF || Rank[k]>Rank[i])
						break;
				}
				if (j!=f->numParents)
					break;
			}
		}
		if (i!=numVariables)
			return FALSE;
		return TRUE;
	}*/

	//we are adding one parent to the family FamSet (which contains d parents)
	//BestCandidates contaont the numCandidates best families found so far
	//we have to update this array with every possible augmentation of FamSet
	/*void	RecBuildFamilies(Family *FamPool, int *parentSet, int v, int d, double * BestScores, int numCandidates
							double **SetValues, BOOL** SetReached)
	{

		int v2;
		int pos = 0;
		int k;

		for ( v2 = d ; v2 > 0; v2--)
			FamSet[v2] = FamSet[v2-1];

		for ( v2 = 0 ; v2 <numVariables-1; v2++)
		{
			if ( parentSet[pos+1] < v2)
			{
				parentSet[pos+1] = parentSet[pos];
				pos++;
			}

			//pos is the position where we should put v2
			parentSet[pos] = v2;

			for ( k = 0; k < d ; k++)
			{
				if ( parentSet[k] < v)
				{
					childrenSet[k] = parentSet2[k] = parentSet[k];
					p = k;
				}
				else
				{
					childrenSet[k+1] = parentSet2[k] = parentSet[k]+1;
				}
	
			}
		
			childrenSet[p+1] = v; 

			childrenPtr = SetToPtr(numVariables  , d+1 , childrenSet);
			parentPtr = SetToPtr(numVariables - 1 , d, parentSet);
			parentPtr2 = SetToPtr(numVariables , d, parentSet2);

			if (!SetReached[d+1][childrenPtr])
			{

				for ( k = 0; k < d+1 ; k++)
					Domains[k] = stats->GetVariableDomain( childrenSet[k] );

				count.InitCounter(d+1, childrenSet, Domains, stats, VIRTUAL_COUNT, Tree);

				SetValues[d+1][childrenPtr] = count.GetBayesianValue();
				
				SetReached[d+1][childrenPtr] = TRUE;

				SetReachedCount++;
			}
			
			if (!SetReached[d][parentPtr2])
			{

				for ( k = 0; k < d ; k++)
					Domains[k] = stats->GetVariableDomain( parentSet2[k] );

				count.InitCounter(d, parentSet2, Domains, stats, VIRTUAL_COUNT, Tree);

				SetValues[d][parentPtr2] = count.GetBayesianValue();
				
				SetReached[d][parentPtr2] = TRUE;

				SetReachedCount++;
			}

			//the score of the current family
			score = SetValues[d+1][childrenPtr] - SetValues[d][parentPtr2];
			
			for (i = numCandidates-1 ; i >=0 ; i--)
			{
				if (BestScores[i] > score)
					break;

			}

			if (i!=numCandidates-1)
			{
				//the current Family should be put in position i+1
				
				//keep a pointer to the family with its score

			}
		}

		//iterate thourgh the kept families and find the families with score >= BestScore[numCandidate-1]
		//for these family
		//store them and launch recursively this function on them
					fam.score = score;
					
					//iterate through children to find the max score of max scores of children
//#ifdef ORDER_SPACE
					fam.numParents = d;
					fam.ptr = parentPtr;
					maxscore = SCORE_MIN;

					if ( d != 0)
					{
						//we consider every possible family reduction
						//int p = 0;//we are trying to add an element before the pth parent of parentSet
						for (vv = 0; vv < d; vv++)
						{
							//we are deleting the vth variable from parentSet
							for (v2 = 0; v2 < d; v2++)
							{
								if ( v2 < vv)
									parentSet2[v2] = parentSet[v2];
								else if (v2 > vv)
									parentSet2[v2-1] = parentSet[v2];
							}
							//the kth variable is a possible augmentation

			
							ptr = SetToPtr( numVariables-1, d - 1, parentSet2);//FamScoreLinks[ d ][ parentPtr ][ v ];
							score = MaxScores[ d-1 ][ ptr ];
							score2 = FamilyTree[ d-1 ][ ptr ].score;

							if (score > maxscore)
								maxscore = score;
							if (score2 > maxscore)
								maxscore = score2;

						
						}	
					}

					if (d!=maxParents-1)
						MaxScores[ d ][ parentPtr ] = maxscore;
	
					FamilyTree[ d ][ parentPtr ] = fam;
					if (fam.score > maxscore)
						FamilyList[ v ] [ FamilyCount[v]++ ] = fam;

//#endif



					NextSet(numVariables-1, d , parentSet);



	}*/

	void	quickSort(Family * l, int g, int d)
	{
		int i, m;
		double v;
		Family x;

		if (g<d)
		{
			v = l[g].score;
			m = g;
			for ( i = g + 1; i<=d; i++)
			{
				if (l[i].score > v)
				{
					m++;
					x = l[m];
					l[m] = l[i];
					l[i] = x;
				}
			}
			x = l[m];
			l[m] = l[g];
			l[g] = x;
			quickSort(l, g, m-1);
			quickSort(l,m+1,d);
		}
	}

	/*void FamilyToGraph(int * FamilyPtr, BOOL * Graph)
	{
		int i, p ;

		for ( i = 0; i < numVariables*numVariables; i++)
		{
			Graph[i] = FALSE;
		}

		Family *f;
		for (i = 0; i < numVariables; i ++)
		{
			f = FamilyList[i][FamilyPtr[i]];
			for ( p = 0; p < f->numParents ; p++)
			{
				//if (p==1 && i==1)
				//	printf("error");

				Graph[f->Parents[p]*numVariables+i]=TRUE;
			}
		}

	}*/



	/*void printScore(BOOL *Graph)
	{
		int * parent = new int[maxParents];
		int numParents;
		double score = 0;
		for (int n = 0; n < numVariables ; n++)
		{
			numParents = 0;

			for (int p = 0; p < numVariables; p++)
			{
				if (Graph[p*numVariables+n])
				{
					ASSERT(numParents!=maxParents);
					parent[numParents++] = (p>n)?p-1:p;
				}
			}

			score += FamilyTree[n][numParents][SetToPtr(numVariables - 1, numParents, parent)].score;
			printf("e %f\n",FamilyTree[n][numParents][SetToPtr(numVariables - 1, numParents, parent)].score);


		}

		delete parent;
	
		printf("score=%f\n",score);

	}*/


	double TabuStructureSearch(State *BestState, int l, int n)
	{
		int improvement = 0;

		Operator omax;


		int t;

#ifdef GRAPH_SPACE
		Chain *c;
		State *state = new State(numVariables, l, this);
#endif

#ifdef ORDER_SPACE
		State *state = new State(numVariables, this);
		Operator *tabu = new Operator[l];
		BOOL	*tabuGraph = new BOOL[numVariables*numVariables];
		for ( t = 0; t < numVariables*numVariables; t++)
			tabuGraph[t] = FALSE;		
		for ( t = 0; t < l; t++)
			tabu[t].type = UNDEF;
		Operator o, ostart;
		double maxdelta = 0, delta; 
#endif



		double currentscore, lastscore, maxscore;

		state->CopyStateFrom(BestState);
		currentscore = state->GetScore(maxParents);
		lastscore = currentscore;


		t = 0;

		while (improvement < n)
		{
			
#ifdef GRAPH_SPACE

			c = state->GetBestOperator();

			while (TRUE)
			{
			
				omax = c->Op;

				assert(c->Possible);


				if (state->doOperator(omax,this))
					break;
						
				c = state->GetNextOperator(c);

			}

			maxscore = state->GetScore(maxParents);

			//printf("Doing operator : %d/%d/%d\n",omax.type, omax.from, omax.to);
			//printf("score = %f\n", maxscore);

		
			if (omax.type == 2 && omax.from == 6 && omax.to == 4)
				omax.type = 2;

		
			if (l!=0)
			{
				state->PutTabu(omax,t);
				t = (t+1)%l;

	
			}
#endif

#ifdef ORDER_SPACE
			omax.type = UNDEF;

			ostart = o = state->Sample();

			do
			{
					delta = state->GetDeltaScore(o);

					if (!tabuGraph[o.var2 * numVariables + o.var1] && (omax.type == UNDEF ||delta > maxdelta ))
					{	
							omax = o;
							maxdelta = delta ;
					} 
					
					o = state->Next(o);

			} while ( !o.equalTo(ostart) );

			assert (omax.type != UNDEF);


			state->doOperator(omax,this);

			maxscore = state->GetScore(maxParents);

			if (l!=0)
			{
				o = tabu[t%l];
				if (o.type != UNDEF)
					tabuGraph[ o.var1 * numVariables + o.var2] = FALSE;

				tabuGraph[ omax.var1 * numVariables + omax.var2] = TRUE;

				tabu[t%l] = omax;
				t = (t+1)%l;
	
			}
#endif

			UpdateTime();
			
			if (maxscore > currentscore)
			{
				if (maxscore > maxFoundScore)
				{
					if (!OBS_SUPPRESS_OUTPUT) printf("Score=	%f	LLikelihood=	%f	Time=	%f\n", maxscore/StatsTrain->GetnumSamples(),  state->GetLikelihood()/StatsTest->GetnumSamples(), currentTime);
					maxFoundScore = maxscore;
				}
				improvement = 0;
				BestState->CopyStateFrom(state);
				currentscore = maxscore;
			}
			else
				improvement++;

			lastscore = maxscore;

		}
#ifdef ORDER_SPACE
		delete tabu;
		delete tabuGraph;
#endif
		delete state;

		return currentscore;
	}

	void RandomRestartSearch(State * maxState, double time_limit, int deviation, int tabu_size, int depth)
	{
		Operator o;
		double s = SCORE_MIN,s2;
#ifdef GRAPH_SPACE
		State * state = new State(numVariables, 0, this);
#endif
#ifdef ORDER_SPACE
		State * state = new State(numVariables, this);
#endif

		maxFoundScore = SCORE_MIN;
		state->Init( this );

	
		while ( currentTime < time_limit )
		{

			//printf("Random Restart- %d/%d-----\n",r+1,nb_restart);
			
			s2 = TabuStructureSearch(state,tabu_size,depth);

			if (s<s2)
			{
				maxState->CopyStateFrom(state);
				s = s2;
			}

			//Perform Random Walk
			for (int no = 0; no < deviation; no++)
			{
				do
				{
					o = state->Sample();
				} while (!state->isLegalOp(o,NULL,0,0,maxParents));
				state->doOperator(o,this);
			}
		
		}
 
		delete state;
	}
		
};


class SetFamScores
{
	int	Variable;
	int	FamilySize; //counting the child node

	int		tablesize;
	double	*scores;//score [ r1 + (numNodes - 1) * ( r2 + (numNodes - 2) * ( ... + ( rk )...)) ] 
					//= score of Family with n1,n2,...,nk as parents where ni < ni+1 ni are given forgetting node Variable 
					//  where r1 = n1,  and rk = n2-n1-1 >= 0
public:
	SetFamScores(int var, int familysize, Statistics * stats)
	{
		
		int i;

		//DEBUG
		clock_t start = clock();
		
		Variable = var;
		FamilySize = familysize;

		int parentSize = FamilySize-1;

		int nmax = stats->GetnumVariables();

		//family : old referential of nodes
		int * family = new int[FamilySize];
		//domains of variables
		int * dom = new int[FamilySize];

		//new referencial of nodes -> old referential of nodes
		int	* n = new int[nmax - 1];

		//coordinate of scores table (usual coordinate)
		int * r = new int[parentSize ];

		//coordinate of scores table (usual coordinate)
		//int * r2 = new int[parentSize];

		//DEBUG
		//int * r3 = new int[parentSize];

		int ptr = 0;
		//maximum value of r
		//int *rmax = new int[parentSize ]; 
		//note: the domain depends on the last elements


		for ( i = 0; i < nmax - 1; i++)
		{
			if (i >= var)
				n[i] = i+1;
			else
				n[i] = i;
		}

		//tablesize = c(nmax-1,parentSize) = (nmax-1)!/(nmax - 1 - parentSize)!*parentSize!

		tablesize = Combin[nmax - 1][parentSize];

		for (i = 0; i < parentSize; i++)
		{
			r[i] = i;
			//domr[i] =  rmax;
		}

		family[0] = var;
		dom[0] = stats->GetVariableDomain(var);
		//rmax[0] = nmax - parentSize;

		scores = new double[tablesize];

		int rexp;

		Counter*	c;

		do
		{ 
			//r => Family and domains
			
			//DEBUG
			if ( (ptr & 0x3FF) == 0)
			{
				if (!OBS_SUPPRESS_OUTPUT) printf("%d/%d\n",ptr,tablesize);
			}

			//r2[0] = r[0];
			//family[1] = n[r[0]];
			//dom[1] = stats->GetVariableDomain(family[1]);
			for (int i = 0; i < parentSize ; i++)
			{
				family[ i + 1 ] = n[ r[i] ];
				dom[i + 1] = stats->GetVariableDomain(family[i + 1]);

				//family[i+1]<nmax - (parentSize-i) + 1
				//=> r[i] = family[i+1] - family[i] - 1 < nmax - (parentSize-i) -family[i] 
				//rmax[i] = nmax - r2[i-1] - (parentSize-i) - 1;
			}
			
			//ASSERT( ptr == SetToPtr(nmax - 1, parentSize, r2));

			//PtrToSet(ptr, nmax - 1, parentSize, r3);
			//for (int t = 0; t<parentSize ; t++)
			//	ASSERT( r3[t] == r2[t] );

			c = new Counter(FamilySize, family, dom, stats, 1);

			scores[ptr] = c->GetScore(var);

			delete	c;

			ptr++;

			//rexp = NextAssignmentRight(parentSize, r, rmax); 
			
			rexp = NextSet(nmax - 1, parentSize, r);

			//rexp = PtrToSet(ptr, nmax - 1, parentSize, r3);
			//for (t = 0; t<parentSize ; t++)
			//	ASSERT( r3[t] == r2[t] );


		} while (rexp >=0);

		delete family;
		delete dom;
		delete r;
		delete n;

		//DEBUG		duration = (double)(finish - start) / CLOCKS_PER_SEC;
		if (!OBS_SUPPRESS_OUTPUT) printf( "%f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC );

	}

	~SetFamScores()
	{
		delete scores;
	}

};


class CClasses  
{
 public:
  CClasses() { }
  virtual ~CClasses() { }

};

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

void	Factor::SetScope(int* vars, Network * net)
	{
		int i,s;

		delete Variables;
		delete Domains;
		delete table;

		Variables = new int[numVariables];
		memcpy(Variables,vars,sizeof(int)*numVariables);

		Domains = new int[numVariables];

		for (i = 0, s = 1; i < numVariables;i++)
		{
			Domains[i] = net->GetNode(vars[i])->GetVariableDomain();
			s *= Domains[i];
		}

		tablesize = s;
		table = new double[tablesize];

	
	};

void Node::LearnCPT(Statistics * stats, double initValue)
{
	int		*scope;
	int		*a;
	Counter *counter;
	double	*valuecount;

	double ct;
	int j,k;
	int d = GetVariableDomain();

	//construct the 2 counter and assign values to the corresponding CPTs
	scope = new int[numVariables];
	GetScope(scope);
	counter = new Counter(this, stats, initValue);
	valuecount = new double[d];
	a = new int[numVariables];

	for ( k = 0 ; k< numVariables; k++)
		a[k] = 0;

	//iterates through assignement
	do
	{
		ct = 0;

		for ( k = 0 ; k < d; k++)
		{
			a[0] = k;
			valuecount[k] = counter->GetCount(a);
			ct += valuecount[k];
		}

		for (k = 0 ; k < d; k++)
		{
			a[0] = k;
			SetValue(valuecount[k]/ct,a);
		}

		j = NextAssignment(numVariables, a, Domains);
		

	} while (j < numVariables);

	delete counter;
	delete scope;
	delete valuecount;
	delete[] a;


}


void	Factor::printComparison(Factor* factor)
{
	int i;

	if (numVariables != factor->GetnumVariables())
	{
		if (!OBS_SUPPRESS_OUTPUT) printf("Factors disagree on number of variables.\n");
		return;
	}

	for ( i = 0; i < numVariables ; i++)
	{
		if (Variables[i] != factor->GetVariable(i))
		{
			if (!OBS_SUPPRESS_OUTPUT) printf("Factors disagree on variable %d.\n",i);
			return;
		}

		if (Domains[i] != factor->GetDomain(i))
		{
			if (!OBS_SUPPRESS_OUTPUT) printf("Factors disagree on domain of variable %d.\n",i);
			return;
		}
	}

	ASSERT(tablesize == factor->GetSize());

	int * a = new int[numVariables];
	
	for (i = 0; i < numVariables; i++)
		a[i] = 0;

	do
	{
		if (!OBS_SUPPRESS_OUTPUT) printf("%f - %f\n",GetValue(a),factor->GetValue(a));

		i = NextAssignment(numVariables,a,Domains);

	} while (i < numVariables);

}

void Factor::multiply(Factor* factor)
{
	int		numv1 = GetnumVariables();
	int		numv2 = factor->GetnumVariables();
	int*	scope2 = new int[numv2];

	factor->GetScope(scope2);

	int*	scope = new int[numv1 + numv2];
	int*	domains = new int[numv1 + numv2];
	int		newtablesize = 1;
	int		numv = numv1;

	GetScope(scope);
	GetDomains(domains);
	newtablesize = GetSize();

	int*	posc = new int[numv2];//common variables
	int*	pose = new int[numv2];//varaibles exclusively in factor 2
	int*	domc = new int[numv2];
	int*	dome = new int[numv2];
	int		numvc = 0;
	int		numve = 0;
	
	int v,v2,k,ptr,nptr;

	for ( v = 0; v < numv2 ; v++)
	{
		for ( v2 = 0; v2 < numVariables ; v2++)
		{
			if (scope[v2] == scope2[v])
				break;
		}
		if (v2 == numVariables)//new variable
		{
			scope[numv] = scope2[v]; 
			domains[numv] = factor->GetDomain(v);
			newtablesize *= domains[numv];
			numv++;


			pose[numve] = v;
			dome[numve] = factor->GetDomain(v);
			numve++;
		}
		else
		{
			//common variable
			posc[numvc] = v;
			domc[numvc] = factor->GetDomain(v);
			numvc++;
		}
	}
	
	double* newtable = new double[newtablesize];
	
	int* asse = new int[numve];
	int* assc = new int[numvc];
	int* ass2 = new int[numv2];

	for ( k = 0; k < numvc; k++)
		assc[k] = 0;

	//TODO : 
	//Optimisation possible here
	for ( ptr = 0, nptr = 0; ptr < tablesize ; ptr++)
	{
		//we have table[ptr] : the value of the first factor

		//build partial assignment corresponding to ptr of the second factor 
		//easy just add 1 more
		NextAssignment(numvc,assc,domc);

		for ( v2 = 0; v2 < numve; v2++)
			asse[v2] = 0;
		
		//loop on the rest of the assignment of the second factor
		do 
		{
			//combine two assignements
			for ( v2 = 0; v2 < numve ; v2 ++)
				ass2[pose[v2]] = asse[v2];

			for ( v2 = 0; v2 < numvc ; v2 ++)
				ass2[posc[v2]] = assc[v2];
			
			//	 multiply the result to have final value
			newtable[nptr] = table[ptr] * factor->GetValue(ass2);

			nptr ++;
			v2 = NextAssignment(numve, asse, dome);

		} while (v2 < numve);

		
	}
	
	delete	Variables;
	delete	Domains;
	delete	table;
	delete	pose;
	delete	posc;
	delete	dome;
	delete	domc;
	delete	asse;
	delete	assc;
	delete	asse;

	//realloc scope domains,...
	numVariables = numv;
	Variables = (int*) realloc(scope, sizeof(int) * numv);
	Domains = (int*) realloc(domains, sizeof(int) * numv);
	tablesize = newtablesize;
	table = newtable;
}

void Factor::divide(Factor* factor)
{
	int		numv2 = factor->GetnumVariables();
	int*	scope2 = new int[numv2];
	int*	dom2 = new int[numv2];

	factor->GetScope(scope2);
	factor->GetDomains(dom2);

	int*	pos2 = new int[numv2];//varaibles exclusively in factor 2
	int vc= 0;

	int v,v2,k,ptr,nptr;

	for ( v = 0; v < numv2 ; v++)
	{
		for ( v2 = 0; v2 < numVariables ; v2++)
		{
			if (Variables[v2] == scope2[v])
				break;
		}
		ASSERT(v2 != numVariables);

		//common variable
		pos2[vc] = v;
		vc++;
	}
	
	int*	ass1 = new int[numVariables];
	int*	ass2 = new int[numv2];
	double	val;

	for ( k = 0; k < numVariables; k++)
		ass1[k] = 0;

	for ( ptr = 0, nptr = 0; ptr < tablesize ; ptr++)
	{
		//we have table[ptr] : the value of the first factor

		for ( v2 = 0; v2 < numv2; v2++)
			ass2[pos2[v2]] = ass1[v2];
		
		val = factor->GetValue(ass2);
		if (val == 0)
		{
			ASSERT( table[ptr] == 0 );
		}
		else
			table[ptr] /= val;

		NextAssignment(numVariables, ass1, Domains);
		
	}
	
	delete	scope2;
	delete	dom2;
	delete	pos2;
	delete	ass1;
	delete	ass2;

}

void Factor::sumOut(int numv2, int* scope2 )
{
	int ve = 0;
	int	v,v2,k,ptr,nptr;
	int * pose = new int[numVariables];

	for ( v = 0; v < numVariables ; v++)
	{
		for ( v2 = 0; v2 < numv2 ; v2++)
		{
			if (Variables[v] == scope2[v2])
				break;
		}

		if (v2 == numVariables)
		//the variable is not in the scope of the factor
		{
			//exclusive variable of the factor
			pose[ve] = v;
			ve++;	
		}

	}

	int		oldnumVar = numVariables;
	int*	oldVariables = Variables;
	int*	oldDomains = Domains;
	
	numVariables = ve;
	Variables = new int[numVariables];
	Domains = new int[numVariables];
	
	int		oldtablesize = tablesize;

	tablesize = 1;
	for ( v = 0; v < ve ; v++)
	{
		Variables[v]=oldVariables[pose[v]];
		Domains[v]=oldDomains[pose[v]];
		tablesize *= Domains[v];
	}
	
	double* oldtable = table;
	table = new double[tablesize];

	//we initialize the new table
	for ( k = 0; k < tablesize; k++)
		table[k] = 0;

	int*	ass1 = new int[oldnumVar];
	int*	ass2 = new int[numVariables];

	for (k = 0; k < oldnumVar; k++)
		ass1[k] = 0;

	for ( ptr = 0, nptr = 0; ptr < oldtablesize ; ptr++)
	{
		//we have oldtable[ptr] : the old value of the factor

		for (int v2 = 0; v2 < numVariables; v2++)
			ass2[pose[v2]] = ass1[v2];
		
		AddValue(oldtable[ptr],ass2);

		NextAssignment(oldnumVar, ass1, oldDomains);
		
	}
	
	delete	oldVariables;
	delete	oldDomains;
	delete	oldtable;
	delete	pose;
	delete	ass1;
	delete	ass2;

}



double	Node::GetScore(Statistics *stats, double initValue)
{
	int		*scope;
	Counter *counter;

	double score = 0;
	//int d = GetVariableDomain();

	//construct the counter
	scope = new int[numVariables];
	GetScope(scope);
	counter = new Counter(this, stats, initValue);
	
	score = counter->GetScore(scope[0]);
	//printf("n %f\n", score);

	delete counter;
	delete scope;

	return score;

}



#ifdef ORDER_SPACE
	
void State::Init(GreedySearch * search)
	{
          int n, n2, r, p ;

          /*
            // This code seems to just be choosing a random permutation
            // of the variables, but it's doing it in a slow way,
            // so I've written a faster thing to do it below. (Joseph)

		for (p = 0; p < numVariables;p++)
		{
			Ordering[p] = UNDEF;
		}

		for (n = 0; n < numVariables;n++)
		{
			do
			{
				p = (int) (RANDOM * numVariables);
			} while (Ordering[p] != UNDEF);
			
			Rank[n] = p;

			Ordering[p] = n;
		}
          */
          for (n = 0; n < numVariables; n++)
            Ordering[n] = n;
          for (n = 0; n < numVariables; n++) {
            // Choose an index in [n, numVariables-1]
            p = (int) (RANDOM * (numVariables - n));
            p = (p < 0 ? n : p + n);
            // Swap variables Ordering[n] and Ordering[p] and
            //  fix Ordering[p] in position n
            Rank[Ordering[p]] = n;
            int tmpint = Ordering[p];
            Ordering[p] = Ordering[n];
            Ordering[n] = tmpint;
          }

		FamilyList = &search->FamilyList;
		for (n = 0; n < numVariables;n++)
		{
			for ( p = 0; p < FamilyCount[n] ; p++)
			{
				ValidityCount[n][p] = search->FamilyList[n][p].numParents;
				if (ValidityCount[n][p] == 0)
					BestFamilyPtr[ n ] = p;
			}
			
			for ( r = 0; r < Rank[n]; r++ )
			{
				n2 = Ordering[r];
				UpdateFamPtrDown(n, n2, search->InFamilyCount, search->InFamilyPtr);
			}

		}

		for (n = 0; n < numVariables - 1; n++)
			deltaScores[n] = GetPotentialDeltaUp( Ordering[n+1], Ordering[n], search->InFamilyCount, search->InFamilyPtr)
								+ GetPotentialDeltaDown( Ordering[n], Ordering[n+1], search->InFamilyCount, search->InFamilyPtr);


	}


	State::State(int n, GreedySearch *search)
	{
		numVariables=n;
		numCandidates=search->numCandidates;
		Candidates = search->Candidates;
		Rank = new int[numVariables];
		Ordering = new int[numVariables];
		//FamilyList = new Family*[numVariables];
		ValidityCount = new int*[numVariables];
		FamilyCount = new int[numVariables];
		memcpy(FamilyCount, search->FamilyCount, sizeof(int)*numVariables);
		deltaScores = new double[numVariables-1];
		//tabuGraph = new BOOL[numVariables*numVariables];
		FamilyList = NULL;

		for (int v=0; v < numVariables; v++)
		{
			//FamilyList[v] = new Family[ FamilyCount[v] ];
			ValidityCount[v] = new int[ FamilyCount[v] ];
		}
		
		BestFamilyPtr = new int[numVariables];

	}

	void State::doOperator(Operator o, GreedySearch * search)
	{
 		ASSERT(o.type == REVERSE_NODE);
		ASSERT(o.pos == Rank[o.var1]);

		/*printf("Operator %d %d->%d\n", var, from, to);

		for (int va = 0;  va < s->numVariables ; va++)
			printf("%d [%d]\n",va,s->Ordering[va]);*/

		Ordering[o.pos] = o.var2;
		Ordering[o.pos+1] = o.var1;
		Rank[o.var1] = o.pos+1;
		Rank[o.var2] = o.pos;


		UpdateFamPtrDown(o.var1, o.var2,  search->InFamilyCount, search->InFamilyPtr);
		UpdateFamPtrUp(o.var2, o.var1,  search->InFamilyCount, search->InFamilyPtr);


		UpdateDeltaScore(o.pos , search->InFamilyCount, search->InFamilyPtr);
		//DEBUG
		//GetScore(FamScores, FamilyCount, FamilyList, maxParents);
		/*for ( va = 0;  va < s->numVariables ; va++)
			printf("%d [%d]\n",va,s->Ordering[va]);*/
	}

	void State::undoOperator(Operator o, GreedySearch * search)
	{
		ASSERT(o.type == REVERSE_NODE);
		/*printf("UnOperator %d %d->%d\n", var, from, to);

		for (int va = 0;  va < s->numVariables ; va++)
			printf("%d [%d]\n",va,s->Ordering[va]);*/

		Ordering[o.pos] = o.var1;
		Ordering[o.pos+1] = o.var2;
		Rank[o.var1] = o.pos;
		Rank[o.var2] = o.pos+1;

		UpdateFamPtrUp(o.var1, o.var2,  search->InFamilyCount, search->InFamilyPtr);
		UpdateFamPtrDown(o.var2, o.var1,  search->InFamilyCount, search->InFamilyPtr);

		UpdateDeltaScore(o.pos , search->InFamilyCount, search->InFamilyPtr);		
		
		/*for (va = 0;  va < s->numVariables ; va++)
			printf("%d [%d]\n",va,s->Ordering[va]);*/
	}

	void State::UpdateDeltaScore( int n, int** InFamilyCount, int ***InFamilyPtr)
	{
		deltaScores[n] = -deltaScores[n];
		if (n != 0)
			deltaScores[n - 1] = GetPotentialDeltaUp( Ordering[n], Ordering[n-1], InFamilyCount, InFamilyPtr)
								+ GetPotentialDeltaDown( Ordering[n-1], Ordering[n], InFamilyCount, InFamilyPtr);
		
		if (n != numVariables - 2)
			deltaScores[n + 1] = GetPotentialDeltaUp( Ordering[n+2], Ordering[n+1], InFamilyCount, InFamilyPtr)
								+ GetPotentialDeltaDown( Ordering[n+1], Ordering[n+2], InFamilyCount, InFamilyPtr);

	}

	double	State::GetDeltaScore( Operator o)
	{
		return deltaScores[o.pos];
	}

	int State::GetNumOp( Operator o)
	{
		return o.pos;
	}

	Operator State::GetFirstOp()
	{
		Operator o;
		o.type = REVERSE_NODE;
		o.pos = 0;
		o.var1 = Ordering[o.pos];
		o.var2 = Ordering[o.pos+1];
		return o;

	}

	void State::UpdateFamPtrUp(int n, int n2, int** InFamilyCount, int ***InFamilyPtr)
	{
		//if (n==1)
		//	printf("ERROR");

		int p;

		/*for ( p = 0; p < InFamilyCount[n][n2]; p++)
			FamilyList[n][ InFamilyPtr[n][n2][p] ].validCount++;
		
		for (p = BestFamilyPtr[n] ; p < FamilyCount[n]; p++)
		{
			if (FamilyList[n][p].validCount==0)
				break;
		}

		int test = p;*/

		for ( p = 0; p < InFamilyCount[n][n2]; p++)
			ValidityCount[n][ InFamilyPtr[n][n2][p] ]++;
		
		for (p = BestFamilyPtr[n] ; p < FamilyCount[n]; p++)
		{
			if (ValidityCount[n][p]==0)
				break;
		}

		//ASSERT( p == test );
		ASSERT( p!=FamilyCount[n] );
		BestFamilyPtr[ n ] =  p ;
	}

	void State::UpdateFamPtrDown(int n, int n2, int** InFamilyCount, int ***InFamilyPtr)
	{
		int p;

		/*int test = BestFamilyPtr[n];
		for ( p = 0; p < InFamilyCount[n][n2] && InFamilyPtr[n][n2][p] < test ; p++)
		{
			if (--FamilyList[n][ InFamilyPtr[n][n2][p] ].validCount==0)
				test = InFamilyPtr[n][n2][p];
		}

		for (; p < InFamilyCount[n][n2]; p++)
			FamilyList[n][ InFamilyPtr[n][n2][p] ].validCount--;
		*/

		for ( p = 0; p < InFamilyCount[n][n2] && InFamilyPtr[n][n2][p] < BestFamilyPtr[n] ; p++)
		{
			if (--ValidityCount[n][ InFamilyPtr[n][n2][p] ]==0)
				BestFamilyPtr[n]  = InFamilyPtr[n][n2][p];
		}

		for (; p < InFamilyCount[n][n2]; p++)
			ValidityCount[n][ InFamilyPtr[n][n2][p] ]--;

		//ASSERT( test == BestFamilyPtr[n]);

		
	}

	double State::GetPotentialDeltaDown( int n, int n2, int** InFamilyCount, int ***InFamilyPtr)
	{
		int p;

		for ( p = 0; p < InFamilyCount[n][n2] && InFamilyPtr[n][n2][p] < BestFamilyPtr[n] ; p++)
		{
			if (ValidityCount[n][ InFamilyPtr[n][n2][p] ]==1)
			{
				return (*FamilyList)[n][ InFamilyPtr[n][n2][p] ].score - (*FamilyList)[n][ BestFamilyPtr[n] ].score;
			}
		}
		return 0;
	}

	
	double State::GetPotentialDeltaUp( int n, int n2, int** InFamilyCount, int ***InFamilyPtr)
	{

		int p = 0;
		int pb = BestFamilyPtr[n];

		while (TRUE)
		{
			while (p < InFamilyCount[n][n2]  && InFamilyPtr[n][n2][p] < pb )
				p++;
			
			if ( p == InFamilyCount[n][n2] || InFamilyPtr[n][n2][p] > pb )
			{
				return (*FamilyList)[n][ pb ].score - (*FamilyList)[n][ BestFamilyPtr[n] ].score;
			}

			pb++;

			while (ValidityCount[n][pb]!=0)
				pb++;
		}
	}
#endif // #ifdef ORDER_SPACE

#ifdef GRAPH_SPACE
	
	State::State(int n, int l, GreedySearch *search)
	{
		numVariables=n;
		Graph = new BOOL[numVariables*numVariables];
		famScores = new double[numVariables];
		//Connexity = new BOOL[numVariables*numVariables];
		//Connexity2 = new BOOL[numVariables*numVariables];
		//oldConnexity = new BOOL[numVariables*numVariables];
		
		//FamilyTreePtr = &(search->FamilyTree);
		maxParents = search->maxParents ;
		Marked = new int[numVariables];
		//parent = new int[maxParents+1];
		//numParents = new int[numVariables];
		numCandidates = search->numCandidates;
		Candidates = search->Candidates;
		CandidatesNb = search->CandidatesNb;

		deltaScores = new ListOp(this);
		tabuSize = l;
		tabuList = new Operator[l];
		int t;
		for ( t = 0; t < l ; t++)
			tabuList[t].type = UNDEF;
		
		Parents = new ListParent[numVariables];
		for ( t = 0; t < numVariables; t++)
			Parents[t].Init(numVariables);

//#ifdef STATS_SET
		SetReachedCountPtr = &(search->SetReachedCount);
		SetReachedPtr = &(search->SetReached);
		SetValuesPtr = &(search->SetValues);
		LSetValuesPtr = &(search->LSetValues);
		TreeTrain	= search->TreeTrain;
		StatsTrain	= search->StatsTrain;
		TreeTest	= search->TreeTest;
		StatsTest	= search->StatsTest;
		AllDomains = search->StatsTrain->Domains;


		parent = new int[numVariables];
		domains = new int[numVariables];

//#endif

	}

	void State::Init(GreedySearch * search)
	{
		int n,n2,k;

		for ( n = 0; n < numVariables*numVariables;n++)
		{
			Graph[n] = FALSE;
			//Connexity[n] = FALSE;
		}

		for ( n=0; n < numVariables ; n++)
		{
			famScores[n] = GetFamScore(n);//(*FamilyTreePtr)[n][0][0].score;
			//Connexity[n*numVariables+n] = TRUE;
			//numParents[n] = 0;
			
			//for ( n2 = 0; n2 < numVariables ; n2++)
			//{
			//	if (n2!=n)
			for ( k = 0 ; k < numCandidates; k ++)
			{
				
				//n3 = (n2<n)?n2:n2-1;
				n2 = Candidates[n][k];
				//Parents[n].Push(n2);
				Parents[n].Push(k);
				//deltaScores->SetScore(ADD_EDGE*numVariables*numVariables + n*numVariables + n2 , GetFamScore(n) - famScores[n]);
				deltaScores->SetScore(ADD_EDGE*numVariables*numCandidates + n*numCandidates + k , GetFamScore(n) - famScores[n]);
				//Parents[n].Pop(n2);
				Parents[n].Pop(k);
				//}
			//	else
			//		deltaScores->SetInvalid(ADD_EDGE*numVariables*numVariables + n*numVariables + n2 );

				//deltaScores->SetInvalid(DELETE_EDGE*numVariables*numVariables + n*numVariables + n2 );
				//deltaScores->SetInvalid(REVERSE_EDGE*numVariables*numVariables + n*numVariables + n2 );
				deltaScores->SetInvalid(DELETE_EDGE*numVariables*numCandidates + n*numCandidates + k );
				deltaScores->SetInvalid(REVERSE_EDGE*numVariables*numCandidates + n*numCandidates + k );

			}
		}	

	}


ListOp::ListOp(State *s)
{
	
	numTotal = s->GetNumberOp();

	List = new Chain[ numTotal ];

	Operator op = s->GetFirstOp();
	numPossible = 0;
	FirstPossible = 0;
	LastPossible = 0;
	for (int i = 0; i < numTotal; i++)
	{
		List[i].Previous = i-1;
		List[i].Next = i+1;
		List[i].Op = op;
		List[i].Id = i;
		List[i].Possible = false;
		List[i].Tabu = 0;
		List[i].Valid = false;
		op = s->Next(op);
	}

	List[0].Previous = numTotal-1;
	List[numTotal-1].Next = 0;

	numPossible = 0;

	//assert(isValid());

}

#endif // #ifdef GRAPH_SPACE

#include <prl/macros_undef.hpp>

#endif // PRL_ORDER_BASED_SEARCH_CLASSES_HPP

#include <iostream>

#include <boost/array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int.hpp>


#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/random/random.hpp>
#include <sill/inference/parallel/basic_state_manager.hpp>
#include <sill/parallel/pthread_tools.hpp>


// This should come last
#include <sill/macros_def.hpp>


using namespace sill;
typedef factor_graph_model<table_factor>::vertex_type vertex_type;


class writetest:public runnable {
public:
  writetest(basic_state_manager<table_factor> *m, factor_graph_model<table_factor> *fg){
    m_=m;
    fg_=fg;
  }
  void run() {
    for (int i = 0; i < 10000; ++i) {
      foreach(finite_variable* var, fg_->arguments()) {
        vertex_type v(var);
        foreach(vertex_type tbl, fg_->neighbors(v)) {
          table_factor *f = m_->checkout(v,tbl,Writing);
          m_->checkin(v,tbl,f);
          
          f = m_->checkout(tbl,v,Writing);
          m_->checkin(tbl,v,f);
        }
      }
    }
   
  }
  
  basic_state_manager<table_factor> *m_;
  factor_graph_model<table_factor> *fg_;
 
};



class readtest:public runnable {
public:
  readtest(basic_state_manager<table_factor> *m, factor_graph_model<table_factor> *fg){
    m_=m;
    fg_=fg;
  }
  void run() {
    for (int i = 0; i < 10000; ++i) {
      foreach(finite_variable* var, fg_->arguments()) {
        vertex_type v(var);
        foreach(vertex_type tbl, fg_->neighbors(v)) {
          table_factor *f = m_->checkout(v,tbl,Reading);
          m_->checkin(v,tbl,f);
          
          f = m_->checkout(tbl,v,Reading);
          m_->checkin(tbl,v,f);
        }
      }
    }
  }
  
  basic_state_manager<table_factor> *m_;
  factor_graph_model<table_factor> *fg_;
 
};


/**
 * \file message_manager.cpp Mesage Manager test
 */
int main() {

  using boost::array;

  // Random number generator
  boost::mt19937 rng;
  boost::uniform_01<boost::mt19937, double> unif01(rng);


  // Create a universe.
  universe u;

  // Create an empty factor graph
  factor_graph_model<table_factor> fg;
  std::cout << "Empty factory graph: " << std::endl;
  fg.print(std::cout);

  // Create some variables and factors
  std::vector<finite_variable*> x(10);
  for(size_t i = 0; i < x.size(); ++i) x[i] = u.new_finite_variable(""+i, 2);

  // Create some unary factors
  for(size_t i = 0; i < x.size(); ++i) {
    // Create the arguments
    finite_domain arguments;  
    arguments.insert(x[i]);
    // Create the table factor and add it to the factor graph
    fg.add_factor( random_discrete_factor<table_factor>(arguments, rng));
  }
  
  // For every two variables in a chain create a factor
  for(size_t i = 0; i < x.size() - 1; ++i) {
    // Create the arguments
    finite_domain arguments;  
    arguments.insert(x[i]);   arguments.insert(x[i+1]);
    // Create the table factor and add it to the factor graph
    fg.add_factor( random_discrete_factor<table_factor>(arguments, rng));
  }

  basic_state_manager<table_factor> manager(&fg, true);
  
  std::cout << "Basic Test : ";
  std::cout.flush();
  foreach(finite_variable* var, fg.arguments()) {
    vertex_type v(var);
    table_factor *f = manager.checkout_belief(v);
    foreach(vertex_type tbl, fg.neighbors(v)) {
      table_factor *f = manager.checkout(v,tbl,Reading);
      manager.checkin(v,tbl,f);
      
      f = manager.checkout(tbl,v,Reading);
      manager.checkin(tbl,v,f);
    }
    manager.checkin_belief(v,f);
  }
  std::cout<<"Ok\n";
    
  std::cout << "Threaded Test : ";
  std::cout.flush();
  thread_group g;
  g.launch(new writetest(&manager, &fg));
  for (int i = 0; i<5; ++i) {
    g.launch(new readtest(&manager, &fg));
  }
  g.join();
  std::cout<<"Ok\n";
  
  {
    std::cout << "Checking Messages : ";
    
    foreach(finite_variable* var, fg.arguments()) {
      vertex_type v(var);
      table_factor *f = manager.checkout_belief(v);
      foreach(vertex_type tbl, fg.neighbors(v)) {
        table_factor *f = manager.checkout(v,tbl,Reading);
        table_factor uniform(f->arguments(),1);
        uniform = uniform.normalize();
        if (*f != uniform) {
          std::cout<<*f;
          std::cout<<uniform;
          std::cerr << "Message have changed!";
          assert(0);
        }
        manager.checkin(v,tbl,f);
        
        f = manager.checkout(tbl,v,Reading);
        table_factor uniform2(f->arguments(),1);
        uniform2 = uniform2.normalize();
        if (*f != uniform2) {
          std::cout<<*f;
          std::cout<<uniform2;
          std::cerr << "Message have changed!";
          assert(0);
        }
        
        uniform = uniform.normalize();
        manager.checkin(tbl,v,f);
      }
      manager.checkin_belief(v,f);
    }
    std::cout << "Ok\n";
  }
  ///TODO: Schedule consistency checks 
  /*{
    std::cout << "Short Var Schedule Consistency check: ";
    std::pair<finite_variable*,double> p = manager.get_top_variable();
    foreach(const table_factor* tbl, fg.factors(p.first)) {
      table_factor *f = manager.checkout_factor_to_variable(tbl,p.first,Writing);
      finite_assignment a_f;
      a_f[const_cast<finite_variable*>(p.first)] = 0;
      (*f)(a_f)=1000000;
      a_f[const_cast<finite_variable*>(p.first)] = 1;
      (*f)(a_f)=1000000;
      manager.checkin_factor_to_variable(tbl,p.first,f);
    }  
    manager.deactivate(p.first);
    std::pair<finite_variable*,double> p2 = manager.get_top_variable();
    assert(p2.first == p.first);
    manager.deactivate(p2.first);
    
    p=manager.get_top_variable();
    assert(p2.first != p.first);
    manager.deactivate(p.first);
    std::cout << "Ok\n";
  }
  
{
    std::cout << "Short Factor Schedule Consistency check: ";
    std::pair<const table_factor*,double> p = manager.get_top_factor();
    foreach(finite_variable* tbl, p.first->arguments()) {
      table_factor *f = manager.checkout_variable_to_factor(tbl,p.first,Writing);
      (*f)(1,1)=100000;
      (*f)(0,0)=100000;
      manager.checkin_variable_to_factor(tbl,p.first,f);
    }  
    manager.deactivate(p.first);
    std::pair<const table_factor*,double> p2 = manager.get_top_factor();
    assert(p2.first == p.first);
    manager.deactivate(p2.first);
    
    p=manager.get_top_factor();
    assert(p2.first != p.first);
    manager.deactivate(p.first);
    std::cout << "Ok\n";
  }  */
}

#include <iostream>

#include <boost/array.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/inference/parallel/message_manager.hpp>
#include <sill/parallel/pthread_tools.hpp>


// This should come last
#include <sill/macros_def.hpp>


using namespace sill;


class writetest:public runnable {
public:
  writetest(basic_message_manager<table_factor> *m, factor_graph_model<table_factor> *fg){
    m_=m;
    fg_=fg;
  }
  void run() {
    for (int i = 0; i < 10000; ++i) {
      foreach(const finite_variable* var, fg_->arguments()) {
        foreach(const table_factor* tbl, fg_->factors(var)) {
          table_factor *f = m_->checkout_variable_to_factor(var,tbl,Writing);
          m_->checkin_variable_to_factor(var,tbl,f);
          
          f = m_->checkout_factor_to_variable(tbl,var,Writing);
          m_->checkin_factor_to_variable(tbl,var,f);
        }
      }
    }
   
  }
  
  basic_message_manager<table_factor> *m_;
  factor_graph_model<table_factor> *fg_;
 
};



class readtest:public runnable {
public:
  readtest(basic_message_manager<table_factor> *m, factor_graph_model<table_factor> *fg){
    m_=m;
    fg_=fg;
  }
  void run() {
    for (int i = 0; i < 10000; ++i) {
      foreach(const finite_variable* var, fg_->arguments()) {
        foreach(const table_factor* tbl, fg_->factors(var)) {
          table_factor *f = m_->checkout_variable_to_factor(var,tbl,Reading);
          m_->checkin_variable_to_factor(var,tbl,f);
          
          f = m_->checkout_factor_to_variable(tbl,var,Reading);
          m_->checkin_factor_to_variable(tbl,var,f);
        }
      }
    }
  }
  
  basic_message_manager<table_factor> *m_;
  factor_graph_model<table_factor> *fg_;
 
};


/**
 * \file message_manager.cpp Mesage Manager test
 */
int main() {

  using boost::array;

  // Random number generator
  boost::mt19937 rng;
  uniform_factor_generator gen;

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
    finite_domain arguments;  
    arguments.insert(x[i]);
    fg.add_factor(gen(arguments, rng));
  }
  
  // For every two variables in a chain create a factor
  for(size_t i = 0; i < x.size() - 1; ++i) {
    finite_domain arguments;  
    arguments.insert(x[i]);
    arguments.insert(x[i+1]);
    fg.add_factor(gen(arguments, rng));
  }

  basic_message_manager<table_factor> manager(fg,true);
  
  std::cout << "Basic Test : ";
  std::cout.flush();
  foreach(const finite_variable* var, fg.arguments()) {
    table_factor *f = manager.checkout_belief(var);
    foreach(const table_factor* tbl, fg.factors(var)) {
      table_factor *f = manager.checkout_variable_to_factor(var,tbl,Reading);
      manager.checkin_variable_to_factor(var,tbl,f);
      
      f = manager.checkout_factor_to_variable(tbl,var,Reading);
      manager.checkin_factor_to_variable(tbl,var,f);
    }
    manager.checkin_belief(var,f);
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
    
    foreach(const finite_variable* var, fg.arguments()) {
      table_factor *f = manager.checkout_belief(var);
      foreach(const table_factor* tbl, fg.factors(var)) {
        table_factor *f = manager.checkout_variable_to_factor(var,tbl,Reading);
        table_factor uniform(f->arguments(),1);
        uniform = uniform.normalize();
        if (*f != uniform) {
          std::cout<<*f;
          std::cout<<uniform;
          std::cerr << "Message have changed!";
          assert(0);
        }
        manager.checkin_variable_to_factor(var,tbl,f);
        
        f = manager.checkout_factor_to_variable(tbl,var,Reading);
        table_factor uniform2(f->arguments(),1);
        uniform2 = uniform2.normalize();
        if (*f != uniform2) {
          std::cout<<*f;
          std::cout<<uniform2;
          std::cerr << "Message have changed!";
          assert(0);
        }
        
        uniform = uniform.normalize();
        manager.checkin_factor_to_variable(tbl,var,f);
      }
      manager.checkin_belief(var,f);
    }
    std::cout << "Ok\n";
  }
  {
    std::cout << "Short Var Schedule Consistency check: ";
    std::pair<const finite_variable*,double> p = manager.get_top_variable();
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
    std::pair<const finite_variable*,double> p2 = manager.get_top_variable();
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
    foreach(const finite_variable* tbl, p.first->arguments()) {
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
  }  
}

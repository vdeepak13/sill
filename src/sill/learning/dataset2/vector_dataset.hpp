        ordering.push_back(table_ptr->weights.size());
        table_ptr->weights.push_back(1.0);
      }
      table_ptr->data.insert(table_ptr->data.end(),
                             nrows * table_ptr->ncols(),
                             -1);
    }

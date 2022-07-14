## **THESIS - ECOMMERCE PRODUCT MATCHING TOOL**

This module uses Rule-Based Classification and Machine Learning Based Classification to decide if any two pairs of products are identical or not. 

To test this module, it is required to input any supplier inventory file.
The file must contain - 

- title 
- default_price_cents
- eans

To run the file, make sure to be in the directory:

`data-management`

To execute -

`python3 tools/wholeseller_inventory_import/product_matching/main.py --supplier_inventory_file`

- supplier_inventory_file: input the path of any supplier inventory file

The Product Matching saved results to local folder named as "results" under product_matching directory
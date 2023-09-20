# ABM
Agents
	Potato Production Areas -Exogenous (AHA)
	Potatoes – Endogenous
	Raw Water – Exogenous (AHA)
	Storage – Exogenous (AHA)
	Logistics - Exogenous (AHA)
	Shippers - Exogenous (AHA)
	Retailers - Exogenous (AHA)
	Consumers – Exogenous (Census Data)

Ceteris Paribus
Note that for potato production we are only considering a single input, raw water.  All other inputs (rain, heat, humidity, fertilizer, evapotranspiration, overwatering, disease) are held constant.
Growth Modeling
General Logistic Growth  
The logistic growth function is a common mathematical model used to represent population growth and other biological processes that follow a characteristic S-shape. It's applied here to describe the potato's growth over time.
To apply the logistic growth model to simulate potato growth, we need to determine the appropriate values for the parameters in the logistic function:
y(t)=  L/〖1+e〗^(-k(t-t_0)) 
Determine the Maximum Size (L): This represents the final weight or size that a potato will reach when fully mature.
Set the Growth Rate Constant (k): The growth rate constant controls how steep the growth curve is. A higher value of k will result in faster growth. 
Time of Maximum Growth Rate (t_0): This represents the inflection point of the S-curve, where growth is fastest. It's typically around the transition between the growth phase and the maturation phase. 0
Integrate into the Step Function:
	At each step, calculate the current time t within the potato's growth cycle.
	Use the logistic growth function to calculate the current size or weight of the potato.
	Update the potato's weight or size accordingly.
	Add some randomness or noise to represent variability among individual potatoes.
Water Supply Effects:
If the potato receives less water than needed, decrease the growth rate constant k, slowing down growth.  If the potato receives more water than needed, this may lead to other effects like reduced quality or diseases.  We will not consider overwatering in this stage of the model.
Harvesting:
Harvest should occur 14 days after maturation to improve skin hardening.
Specific Implementation
Final Weight (L): Representing the final weight as a normally distributed range between 0.2 and 1.0 pounds to account for variation in the final size of the potatoes. Draw from this distribution when initializing each potato agent to give it a unique target size. This will create a population of potatoes with a diverse set of final sizes, reflecting real-world variability.
Growth Rate as a Function of Water (k): The growth rate should be a function of water supply to model the impact of water on growth. Growth rate constant k as a function of the percentage of water needs met:
k= k_(max )×  (water received)/(water required)
Where k_(max )  is the maximum growth rate achievable when all water needs are met. This ensures that the growth rate scales down linearly as water supply decreases, which is a straightforward way to model the effect of water on growth.
Potential Addition of Fertilizers: The framework allows for the incorporation of other factors like fertilizers, extending the model to include a similar mechanism for fertilizers, where meeting fertilizer needs further influences the growth rate constant k, either by modifying it directly or through interactions with water efficiency.  For now, we are holding fertilizer constant.
Note that we are growing potatoes from seeds based on typical planting practices of 24 seed potatoes per acre.  We are excluding the exponential growth where multiple potatoes are produced from one seed, which seems to be about 12-24:1.  The model is just considering a single seed growth at this point, which can be formulated in later iterations.
Water Requirements
In our model, potato plants' irrigation needs are meticulously determined by both their growth stage and the duration spent within that stage. Commencing with the germination phase, no supplemental irrigation is needed post the initial pre-plant hydration. As the plants transition into the vegetative stage, their daily water requirement commences at 0.071 inches and undergoes a weekly increment of the same magnitude until it plateaus at 0.214 inches per day. Subsequently, during the tubering stage, a consistent 0.286 inches of water daily becomes pivotal to emulate an average weekly need ranging between 1.5 to 2.5 inches. This escalates slightly to an average of 0.321 inches daily during full tuberization. The maturation stage witnesses a decremental approach where the initial water requirement is reduced by 10% with each passing week, accommodating the natural decline in hydration needs as the potatoes approach harvesting readiness. This intricate framework ensures that the simulated water provision mirrors the physiological demands of potatoes at different life stages, thereby refining the authenticity of our growth projections.
Field capacity refers to the amount of soil moisture or water content held in soil after excess water has drained away and the rate of downward movement has decreased. This is a unique value for each soil type due to its specific physical properties. It also depends on soil depth, as deeper soils can hold more water.
The rooting depth of mature potato plants is generally between 12 to 30 inches, though most roots are found in the top 12 inches.
To calculate the field capacity for a given acreage:
Soil Type: Different soil types (sandy, loam, clay) have different field capacities.
Effective Rooting Depth: For potatoes, you can use an average value if not given.
Field Capacity Percentage for the Soil Type: This is the percentage of water volume the soil can hold after drainage. It might be something like 20% for sandy soils and 40% for clay soils, as an example.
The formula would be something like:
Field Capacity (acre-feet)=Acres ×Rooting Depth (feet)×Field Capacity %
For 70-80% of this value:
W_0=[0.7,0.8]×Field Capacity
The Snake River Plain in southern Idaho, is one of the primary potato-growing regions in the United States. The soil and climate conditions of this region make it particularly suitable for potato cultivation.
	Soil Type: The predominant soil type in the Snake River Plain is volcanic-ash-influenced soil. These are often called "Andisols" in soil taxonomy. They are well-draining due to the coarse nature of much of the volcanic material.
	Soil Texture: The texture varies, but many of the soils used for potato production are sandy loams or loamy sands. This texture provides good drainage, which is crucial for potato growth to prevent root rot and other diseases.
	Soil pH: Idaho soils tend to be slightly alkaline, with pH values typically ranging from 7.5 to 8.2.
	Organic Matter: The volcanic ash contributes to the relatively high organic matter content in these soils compared to other sandy soils. Organic matter is beneficial for its water-holding capacity and nutrient provision.
	Field Capacity: For sandy loam soils, the typical field capacity might be around 15-20% (by volume).
	Climate and Irrigation: The climate in the Snake River Plain is semi-arid, receiving only about 10-12 inches of precipitation annually. As a result, irrigation is essential for crop production. The primary source of irrigation water is the Snake River, which carries meltwater from the Rocky Mountains.
Given that we're using sandy loam soils in our Idaho-based model, which has a typical field capacity of 15-20% (by volume) and based on research citation which suggests the soil moisture should be about 70-80% of field capacity for pre-plant irrigation (https://cropwatch.unl.edu/potato/plant_growth), let's compute the initial water needs.
Field capacity denotes the amount of water a soil can hold against gravity. If a sandy loam soil has a field capacity of, say, 18% (taking an average), then 70-80% of that would be:
0.7 × 0.18 = 0.126
to 
0.8 × 0.18 = 0.144
This means the soil should hold 12.6% to 14.4% water by volume for pre-plant irrigation.
To convert this percentage into a water volume, we need to know the soil's depth that's relevant for potato growth. Let's say we're considering the top 1 foot (12 inches) of soil for our potato crop, the typical planting depth for potatoes.

Given that 1 acre-foot is the volume of water required to cover an acre of land to a depth of 1 foot, the water required per acre in acre-feet for pre-plant irrigation is:
W_volume (acre-foot)=Acres ×[0.126:0.144]×〖Soil〗_Depth (feet) 

Area Conversion: Inches to Acre-Feet: If 1 acre receives 1 inch of rain, this means:
1 inch ×1 acre=  1/12  acre foot of rain
Given that 1 foot = 12 inches, hence, 1 inch of rain over 1 acre is 1/12 of an acre-foot.
Water Needs Per Plant: If you have 100 potato plants per acre and the entire acre requires 1/12 acre-foot of water, the water requirement for each potato plant would be:
(1⁄12 acre foot)/100=  1⁄1200  ( acrefoot)⁄plant
So, if there are 100 potato plants on an acre and the field needs 1 inch of water, each potato plant effectively requires 1/1200 acre-foot of water.
This assumes uniform distribution of water, which might not be the case in real-world scenarios due to factors like runoff, soil absorption differences, etc.This is just the math. In practice, irrigation doesn't distribute water perfectly evenly, and local soil conditions, wind, temperature, and other factors can affect how much water each plant actually gets. But for modeling purposes, this is a good starting point.
To convert from inches/day to gallons/day for an individual potato, we need to know the area that each potato plant covers. With that area, we can calculate how many gallons of water are needed to cover that area with water to a given depth in inches.

First, a quick reminder of the conversion:

	1 acre-inch is the amount of water that covers 1 acre to a depth of 1 inch.
	1 acre-foot is 12 acre-inches.
	1 acre-foot is equivalent to approximately 325,851 gallons.
	So, 1 acre-inch is equivalent to approximately 325,851/12  or 27,154 gallons.
Now, let's determine the water needs for each potato in gallons/day:

Determine the area each potato plant occupies. If we know the number of potato plants per acre, then:
Area per potato (acres)=  1/(plants⁄acre)= A_p

For any given water need in inches x, the equivalent water need for 1 potato in gallons is:
gallons⁄potato=x ×A_p  ×27,154

Potato Water Needs: Inches to Gallons Conversion
Potato Area Estimate:
We guessed each potato plant uses about 1 square foot of space. This is just our best guess based on some quick observations.
Turning Inches into Gallons:
Research has talked about water needs in terms of inches/day. But we needed to think about it in gallons/day.
One inch of water over an acre is 27,154 gallons. Since an acre is 43,560 square feet, we can figure out the gallons for just 1 square foot.
Updating Our Potato Water Needs Function:
Updated our function that tells us the water needs of a potato. Now, it doesn't just tell us inches/day but converts that to gallons/day using our conversion from point 2.

Introduction
The intricacies of modern supply chains demand sophisticated tools for analysis and optimization. With an emphasis on the agricultural sector, specifically the potato supply chain, this model serves as a vital instrument in understanding how various entities interact within the supply chain ecosystem and how they can be affected by disruptions such as droughts or cyber-attacks.
Model Overview
Architecture
The model is structured around a central class, Potato Supply Chain, which oversees the initiation and interaction of various agent types: Potato Production Area, Potato, Raw Water, Storage, Logistics, Shippers, Retailers and . This class also orchestrates the dynamics of the system, simulating disruptions and gathering data for analysis.
Agent Descriptions
Potato Production Area Agent:
The Potato Production Area class represents a potato production area in the simulation model. Each instance of this class represents a farm or production area from the AHA Core dataset, with certain characteristics such as location (latitude and longitude), owner, state, and the capacity to hold a number of potato agents. It interacts with a water supply agent (Raw Water Agent) to meet its water requirements for various processes like pre-plant irrigation and to provide water to the individual potato agents it holds.
This agent class serves as a significant component in a larger simulation model representing the operations of a potato farm. It includes methods to simulate processes like irrigation, planting, growth, and harvest, and interacts with other agents in the model, such as water supply agents and storage facilities. It maintains various attributes to track the state and properties of the production area over time, supporting a dynamic simulation of a potato farming system.
Potato Agent:
The Potato agent, a subclass of Base Agent, models the life cycle and growing dynamics of a potato plant in a simulated environment. This class encapsulates various attributes and methods that depict the physiological processes and environmental interactions experienced by a potato plant throughout its growth stages from germination to harvest.
This class, endowed with stochastic elements like random growth multipliers and time stages, offers a realistic simulation of a potato's life cycle accounting for natural variability. It integrates the dynamics of water needs across different growth stages, effectively simulating the physiological changes a potato undergoes during its development.
The agent has been constructed to communicate effectively with the Potato Production Area instance, thus aligning individual growth dynamics with larger production area metrics and behaviors. The explicit representation of water stress effects, drought thresholds, and resultant mortality enhance the realism of the simulation.
The Potato agent serves as a highly detailed and scientifically grounded representation of a potato plant's life cycle within a simulation environment, capable of interacting with broader production area dynamics. It can be a crucial component in a larger simulation system aimed at researching and analyzing potato cultivation, offering insights and data that can inform agricultural strategies and optimizations at a graduate research level. It could potentially be used for in-depth studies focusing on crop optimization, sustainable farming practices, and agricultural economics, where a detailed understanding of individual crop dynamics is essential.
Raw Water Agent:
The Raw Water Agent plays a pivotal role in facilitating the dynamics of the potato supply chain, representing the water sources which are integral to the functioning and sustainability of the potato production areas (PPAs).
The Raw Water agent represents sources of raw water essential for potato production. This agent maintains attributes such as water capacity, facility type, and location. Furthermore, it incorporates methods to simulate disruptions in the water supply, either through droughts, affecting a percentage of the water capacity, or full disruptions lasting a number of days.
The Raw Water Agents in this simulation class serve as crucial nodes in the potato supply chain, representing vital resources that underpin the production processes. They are central to modeling spatial dependencies, resource utilization, and disruption dynamics within the supply chain, highlighting the importance of water resources in agricultural supply chain simulations and the potential vulnerabilities introduced by their dependencies.
Storage Agent
In the intricate network of modern supply chain management, the Storage Agent serves as a cornerstone, embodying the principles of efficiency, sustainability, and meticulous resource stewardship. This agent operates as the central repository, harboring commodities while overseeing the balance between capacity and current volume. 
Attributes
1. Name: This attribute imputes the UUID from AHA to each Storage Agent, facilitating easy reference and distinction amidst a network of other agents.
 2. Location: Representing the geographical coordinates, this attribute is pivotal in calculating the logistical dynamics, including transportation cost and time.
3. Capacity: A critical metric denoting the maximum volume the storage facility can accommodate.
4. Current Volume: A dynamic attribute reflecting the existing stockpile within the storage facility. It serves as a critical factor in supply chain decisions, impacting bidding strategies and delivery schedules.
5. Shrinkage Rate: This attribute encapsulates the rate of reduction in commodity volume over time, providing an accurate reflection of the real-time inventory and aiding in the precise management of storage operations.
Methods
1. apply_shrinkage(): This method simulates the known shrinkage that occurs in long term potato storage facilities.
2. notify_logistics(): A method that ensures information flow within the network by notifying the Logistics Agent of the existing inventory levels. 

3. step(): A periodic method that orchestrates the sequential operation of the apply_shrinkage and notify_logistics methods, thereby facilitating a smooth and coordinated workflow, essential in maintaining the equilibrium in supply chain dynamics.
Operational Flow
The Storage Agent operates within a well-defined cyclical process characterized by the following steps:
	Inventory Management: Initiated with an accurate assessment of the current volume of commodities in the storage, this phase ensures that the storage operations align with the pre-defined capacity and shrinkage rate attributes.

	Shrinkage Application: Subsequently, the agent meticulously applies the shrinkage rate, thus adjusting the current volume to reflect the real-time inventory status, a vital step in maintaining the sustainability of the operations.

	Logistical Notification: In the succeeding phase, the agent engages in a proactive dialogue with the Logistics Agent, communicating the updated inventory levels to foster an environment of collaboration and informed decision-making.

	Cyclical Coordination: At the core of its operations, the agent adheres to a cyclical workflow, orchestrated through the step method, which ensures a seamless and coordinated operation, fostering a dynamic yet sustainable supply chain network.

In the grand scheme of supply chain management, the Storage Agent emerges as an entity of strategic significance, fostering an environment characterized by sustainability, efficiency, and meticulous resource management. It operates as a critical node, facilitating the seamless flow of commodities whilst adhering to principles of prudent resource stewardship and collaboration, thereby nurturing a supply chain ecosystem that is both resilient and sustainable.
Logistics Agent
In the rapidly evolving supply chain landscape, the Logistics Agent emerges as a pivotal entity, adept at orchestrating complex interactions between retailers and storage agents within a perfectly competitive market structure. This agent is imbued with functionalities that are meticulously crafted to foster efficiency and economic viability. 
Attributes
1. Commodity Base Price: This attribute encapsulates the baseline price delineated by the storage agent, serving as a quintessential reference in bid evaluations. It is pivotal in discerning the equilibrium between supply and demand, thereby fostering a competitive market dynamic.
   
2. Available Supply: An attribute representing the current stockpile accessible at the storage agent, this is vital in determining the Logistics Agent's capacity to fulfill incoming bids. A prudent management of this attribute ensures a seamless supply chain, preventing bottlenecks and shortages.
   
3. Delivery Cost per Unit Distance: A pivotal parameter, it assists in computing the logistical costs predicated on the distance to the retailer. This attribute is instrumental in fostering cost-effective strategies and optimizing logistical operations.

Methods

1. Receive Bid (bid): This method epitomizes the initial phase of interaction between the retailer and logistics agent, whereby bids from retailers are assimilated and stored meticulously in a bid pool for subsequent evaluation.

2. Evaluate Bids(): At the heart of the Logistics Agent's operations, this method scrutinizes each bid based on a complex criteria matrix encompassing price and volume, juxtaposed with the baseline commodity price. This method leverages analytical acumen to sort bids in a hierarchical manner, prioritizing by price and subsequently by volume.

3. Calculate Delivery Cost(distance): An astute method that determines the logistical expenditures based on the geographical separation between the logistics agent and retailer. This method is integral in delineating the cost dynamics and ensuring the profitability of each transaction.

4. Propose Counteroffer(bid): This method is a testament to the Logistics Agent's strategic foresight, wherein counteroffers are crafted and dispatched based on the analytical evaluation of incoming bids and the existing supply dynamics at the storage agent.

5. Finalize Deals(): Marking the culmination of the bid negotiation process, this method effectively finalizes agreements, cognizant of the supply limitations, and initiates the delivery proceedings, thereby ensuring a seamless transition from bid negotiation to delivery execution.

Operational Flow

The operational cycle of the Logistics Agent is characterized by a series of intricately connected steps:
	Commencement is marked by retailers dispatching their bids, which are stored in a bid pool.
	As the cycle progresses, a comprehensive evaluation of the bids is undertaken, factoring in parameters such as price and volume and the logistical costs dictated by the distance to the retailer.
	Drawing upon strategic insights, the agent either ratifies the most favorable bids or proposes counteroffers, initiating a negotiation cycle with potential for further counteroffers from retailers.
	This is followed by a phase of negotiation, where counteroffers are either accepted or renegotiated, fostering a dynamic and interactive bidding environment.
	The cycle reaches fruition with the finalization of deals, spearheading the initiation of the delivery process, thereby encapsulating a holistic and efficient operational flow.

In this construct, the Logistics Agent emerges as a strategic entity, proficient in fostering market dynamics that are both competitive and efficient, with an acute focus on optimizing both economic viability and logistical efficiency. Its operations are a testimony to a meticulous blend of strategic foresight and analytical prowess, steering the market towards an equilibrium that epitomizes efficiency and competitiveness.
Disruption Simulation
The model integrates a complex disruption simulation subsystem, capable of generating random disruptions affecting the potato production areas and their nearest neighbors. These functions can be used to simulate various scenarios including droughts that disrupt the supply of water from year to year or cyber-attacks that randomly disrupt water supply.
Conclusion
The Potato Supply Chain model is a comprehensive representation of a potato supply chain model where different agents - raw water sources, production areas, and storage facilities - interact and influence each other over time. It incorporates various real-world data and dynamic complexities, such as geographical proximity, resource dependencies, and potential disruptions, offering a versatile platform to study and analyze the dynamics and vulnerabilities of a potato supply chain from various angles including resource management, disruption response, and overall performance tracking through data collection.


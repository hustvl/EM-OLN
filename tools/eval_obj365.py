import sys
sys.path.append('/home/users/cheng03.wang/project/acmmm/object_localization_network')

from pycocotools.coco import COCO
import pycocotools
import numpy as np
from mmdet.datasets.cocoeval_wrappers_acc import COCOEvalWrapper, COCOEvalXclassWrapper

CLASSES = ("Person", "Chair", "Car", "Van", "Formula 1 ", "Sports Car", "SUV", "Bottle", "Cup", "Book", "Boat", "Sailboat", 
                "Ship", "Bench", "Potted Plant", "Vase", "Wine Glass", "Backpack", "Umbrella", "Traffic Light", "Tie", "Bicycle", 
                "Couch", "Bus", "Motorcycle", "Cell Phone", "Truck", "Machinery Vehicle", "Laptop", "Bed", "Horse", "Sink", 
                "Apple", "Knife", "Fork", "Dog", "Spoon", "Clock", "Cow", "Cake", "Sheep", "Keyboard", "Banana", "Baseball Glove", 
                "Airplane", "Helicopter", "Mouse", "Train", "Stop Sign", "Remote", "Refrigerator", "Oven", "Baseball Bat", "Cat", 
                "Broccoli", "Pizza", "Elephant", "Skateboard", "Surfboard", "Donut", "Carrot", "Toilet", "Kite", "Microwave", 
                "Scissors", "Snowboard", "Fire Hydrant", "Zebra", "Giraffe", "Tennis Racket", "Frisbee", "Toothbrush", "Hot dog", 
                "Parking meter", "Sandwich", "Toaster", "Bear", "Dinning Table", "Desk", "Handbag/Satchel", "Hair Dryer", 
                "Wild Bird", "Chicken", "Parrot", "Pigeon", "Duck", "Swan", "Goose", "Bowl/Basin", "Orange/Tangerine", "Luggage", 
                "Briefcase", "Moniter/TV", "Skiboard", "Stuffed Toy", "Baseball", "Basketball", "American Football", "Volleyball", 
                "Golf Ball", "Soccer", "Billards", "Table Tennis ", "Tennis", "Other Balls","Sneakers", "Other Shoes", "Hat", "Lamp", "Glasses", "Street Lights", "Cabinet/shelf", "Bracelet", "Plate", "Picture/Frame", "Helmet", 
                "Gloves", "Storage box", "Leather Shoes", "Flower", "Flag", "Pillow", "Boots", "Microphone", "Necklace", "Ring", "Belt", "Speaker", 
                "Watch", "Trash bin Can", "Slippers", "Stool", "Barrel/bucket", "Sandals", "Bakset", "Drum", "Pen/Pencil", "High Heels", "Guitar", "Carpet", 
                "Bread", "Camera", "Canned", "Traffic cone", "Cymbal", "Lifesaver", "Towel", "Candle", "Awning", "Faucet", "Tent", "Mirror", "Power outlet", 
                "Air Conditioner", "Hockey Stick", "Paddle", "Pickup Truck", "Traffic Sign", "Ballon", "Tripod", "Pot", "Hanger", "Blackboard/Whiteboard", 
                "Napkin", "Other Fish", "Toiletry", "Tomato", "Lantern", "Fan", "Green Vegetables", "Pumpkin", "Nightstand", "Tea pot", "Telephone", 
                "Trolley", "Head Phone", "Dessert", "Scooter", "Stroller", "Crane", "Lemon", "Surveillance Camera", "Jug", "Piano", "Gun", 
                "Skating and Skiing shoes", "Gas stove", "Bow Tie", "Strawberry", "Shovel", "Pepper", "Computer Box", "Toilet Paper", "Cleaning Products", 
                "Chopsticks", "Cutting/chopping Board", "Coffee Table", "Side Table", "Marker", "Pie", "Ladder", "Cookies", "Radiator", "Grape", "Potato", 
                "Sausage", "Tricycle", "Violin", "Egg", "Fire Extinguisher", "Candy", "Fire Truck", "Converter", "Bathtub", "Wheelchair", "Golf Club", 
                "Cucumber", "Cigar/Cigarette ", "Paint Brush", "Pear", "Heavy Truck", "Hamburger", "Extractor", "Extention Cord", "Tong", "Folder", 
                "earphone", "Mask", "Kettle", "Swing", "Coffee Machine", "Slide", "Carriage", "Onion", "Green beans", "Projector", "Washing Machine/Drying Machine", 
                "Printer", "Watermelon", "Saxophone", "Tissue", "Ice cream", "Hotair ballon", "Cello", "French Fries", "Scale", "Trophy", "Cabbage", "Blender", 
                "Peach", "Rice", "Wallet/Purse", "Deer", "Tape", "Tablet", "Cosmetics", "Trumpet", "Pineapple", "Ambulance", "Mango", "Key", "Hurdle", "Fishing Rod", 
                "Medal", "Flute", "Brush", "Penguin", "Megaphone", "Corn", "Lettuce", "Garlic", "Green Onion", "Nuts", "Speed Limit Sign", "Induction Cooker", "Broom", 
                "Trombone", "Plum", "Rickshaw", "Goldfish", "Kiwi fruit", "Router/modem", "Poker Card", "Shrimp", "Sushi", "Cheese", "Notepaper", "Cherry", "Pliers", 
                "CD", "Pasta", "Hammer", "Cue", "Avocado", "Hamimelon", "Flask", "Mushroon", "Screwdriver", "Soap", "Recorder", "Eggplant", "Board Eraser", "Coconut", 
                "Tape Measur/ Ruler", "Pig", "Showerhead", "Globe", "Chips", "Steak", "Crosswalk Sign", "Stapler", "Campel", "Pomegranate", "Dishwasher", "Crab", 
                "Hoverboard", "Meat ball", "Rice Cooker", "Tuba", "Calculator", "Papaya", "Antelope", "Seal", "Buttefly", "Dumbbell", "Donkey", "Lion", "Urinal", 
                "Dolphin", "Electric Drill", "Egg tart", "Jellyfish", "Treadmill", "Lighter", "Grapefruit", "Game board", "Mop", "Radish", "Baozi", "Target", "French", 
                "Spring Rolls", "Monkey", "Rabbit", "Pencil Case", "Yak", "Red Cabbage", "Binoculars", "Asparagus", "Barbell", "Scallop", "Noddles", "Comb", "Dumpling", 
                "Oyster", "Table Teniis paddle", "Cosmetics Brush/Eyeliner Pencil", "Chainsaw", "Eraser", "Lobster", "Durian", "Okra", "Lipstick", "Cosmetics Mirror", "Curling")
COCO_CLASSES = ("Person", "Chair", "Car", "Van", "Formula 1 ", "Sports Car", "SUV", "Bottle", "Cup", "Book", "Boat", "Sailboat", 
                    "Ship", "Bench", "Potted Plant", "Vase", "Wine Glass", "Backpack", "Umbrella", "Traffic Light", "Tie", "Bicycle", 
                    "Couch", "Bus", "Motorcycle", "Cell Phone", "Truck", "Machinery Vehicle", "Laptop", "Bed", "Horse", "Sink", 
                    "Apple", "Knife", "Fork", "Dog", "Spoon", "Clock", "Cow", "Cake", "Sheep", "Keyboard", "Banana", "Baseball Glove", 
                    "Airplane", "Helicopter", "Mouse", "Train", "Stop Sign", "Remote", "Refrigerator", "Oven", "Baseball Bat", "Cat", 
                    "Broccoli", "Pizza", "Elephant", "Skateboard", "Surfboard", "Donut", "Carrot", "Toilet", "Kite", "Microwave", 
                    "Scissors", "Snowboard", "Fire Hydrant", "Zebra", "Giraffe", "Tennis Racket", "Frisbee", "Toothbrush", "Hot dog", 
                    "Parking meter", "Sandwich", "Toaster", "Bear", "Dinning Table", "Desk", "Handbag/Satchel", "Hair Dryer", 
                    "Wild Bird", "Chicken", "Parrot", "Pigeon", "Duck", "Swan", "Goose", "Bowl/Basin", "Orange/Tangerine", "Luggage", 
                    "Briefcase", "Moniter/TV", "Skiboard", "Stuffed Toy", "Baseball", "Basketball", "American Football", "Volleyball", 
                    "Golf Ball", "Soccer", "Billards", "Table Tennis ", "Tennis", "Other Balls")
NONCOCO_CLASSES = ("Sneakers", "Other Shoes", "Hat", "Lamp", "Glasses", "Street Lights", "Cabinet/shelf", "Bracelet", "Plate", "Picture/Frame", "Helmet", 
                    "Gloves", "Storage box", "Leather Shoes", "Flower", "Flag", "Pillow", "Boots", "Microphone", "Necklace", "Ring", "Belt", "Speaker", 
                    "Watch", "Trash bin Can", "Slippers", "Stool", "Barrel/bucket", "Sandals", "Bakset", "Drum", "Pen/Pencil", "High Heels", "Guitar", "Carpet", 
                    "Bread", "Camera", "Canned", "Traffic cone", "Cymbal", "Lifesaver", "Towel", "Candle", "Awning", "Faucet", "Tent", "Mirror", "Power outlet", 
                    "Air Conditioner", "Hockey Stick", "Paddle", "Pickup Truck", "Traffic Sign", "Ballon", "Tripod", "Pot", "Hanger", "Blackboard/Whiteboard", 
                    "Napkin", "Other Fish", "Toiletry", "Tomato", "Lantern", "Fan", "Green Vegetables", "Pumpkin", "Nightstand", "Tea pot", "Telephone", 
                    "Trolley", "Head Phone", "Dessert", "Scooter", "Stroller", "Crane", "Lemon", "Surveillance Camera", "Jug", "Piano", "Gun", 
                    "Skating and Skiing shoes", "Gas stove", "Bow Tie", "Strawberry", "Shovel", "Pepper", "Computer Box", "Toilet Paper", "Cleaning Products", 
                    "Chopsticks", "Cutting/chopping Board", "Coffee Table", "Side Table", "Marker", "Pie", "Ladder", "Cookies", "Radiator", "Grape", "Potato", 
                    "Sausage", "Tricycle", "Violin", "Egg", "Fire Extinguisher", "Candy", "Fire Truck", "Converter", "Bathtub", "Wheelchair", "Golf Club", 
                    "Cucumber", "Cigar/Cigarette ", "Paint Brush", "Pear", "Heavy Truck", "Hamburger", "Extractor", "Extention Cord", "Tong", "Folder", 
                    "earphone", "Mask", "Kettle", "Swing", "Coffee Machine", "Slide", "Carriage", "Onion", "Green beans", "Projector", "Washing Machine/Drying Machine", 
                    "Printer", "Watermelon", "Saxophone", "Tissue", "Ice cream", "Hotair ballon", "Cello", "French Fries", "Scale", "Trophy", "Cabbage", "Blender", 
                    "Peach", "Rice", "Wallet/Purse", "Deer", "Tape", "Tablet", "Cosmetics", "Trumpet", "Pineapple", "Ambulance", "Mango", "Key", "Hurdle", "Fishing Rod", 
                    "Medal", "Flute", "Brush", "Penguin", "Megaphone", "Corn", "Lettuce", "Garlic", "Green Onion", "Nuts", "Speed Limit Sign", "Induction Cooker", "Broom", 
                    "Trombone", "Plum", "Rickshaw", "Goldfish", "Kiwi fruit", "Router/modem", "Poker Card", "Shrimp", "Sushi", "Cheese", "Notepaper", "Cherry", "Pliers", 
                    "CD", "Pasta", "Hammer", "Cue", "Avocado", "Hamimelon", "Flask", "Mushroon", "Screwdriver", "Soap", "Recorder", "Eggplant", "Board Eraser", "Coconut", 
                    "Tape Measur/ Ruler", "Pig", "Showerhead", "Globe", "Chips", "Steak", "Crosswalk Sign", "Stapler", "Campel", "Pomegranate", "Dishwasher", "Crab", 
                    "Hoverboard", "Meat ball", "Rice Cooker", "Tuba", "Calculator", "Papaya", "Antelope", "Seal", "Buttefly", "Dumbbell", "Donkey", "Lion", "Urinal", 
                    "Dolphin", "Electric Drill", "Egg tart", "Jellyfish", "Treadmill", "Lighter", "Grapefruit", "Game board", "Mop", "Radish", "Baozi", "Target", "French", 
                    "Spring Rolls", "Monkey", "Rabbit", "Pencil Case", "Yak", "Red Cabbage", "Binoculars", "Asparagus", "Barbell", "Scallop", "Noddles", "Comb", "Dumpling", 
                    "Oyster", "Table Teniis paddle", "Cosmetics Brush/Eyeliner Pencil", "Chainsaw", "Eraser", "Lobster", "Durian", "Okra", "Lipstick", "Cosmetics Mirror", "Curling")

ann_file = 'path/to/zhiyuan_objv2_val.json'
det_file = 'path/to/det_result.json'
coco_gt = COCO(ann_file)
coco_dt = coco_gt.loadRes(det_file)

eval_cat_ids = coco_gt.get_cat_ids(cat_names=NONCOCO_CLASSES)
train_cat_ids = coco_gt.get_cat_ids(cat_names=COCO_CLASSES)
# import pdb;pdb.set_trace()
for idx, ann in enumerate(coco_gt.dataset['annotations']):
    if ann['category_id'] in eval_cat_ids: # TEST on NonCOCO
        coco_gt.dataset['annotations'][idx]['ignored_split'] = 0
    else:
        coco_gt.dataset['annotations'][idx]['ignored_split'] = 1

cocoEval = COCOEvalXclassWrapper(coco_gt, coco_dt, 'bbox')
cocoEval.params.catIds = coco_gt.get_cat_ids(cat_names=CLASSES)
cocoEval.params.imgIds = coco_gt.get_img_ids()
cocoEval.params.maxDets = (10, 20, 30, 50, 100, 300, 500, 1000)
cocoEval.params.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
cocoEval.params.areaRngLbl = ['all']
cocoEval.params.useCats = 0

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

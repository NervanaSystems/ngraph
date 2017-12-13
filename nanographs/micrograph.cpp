#include <cassert>
#include <memory>
#include <set>
#include <vector>
#include <functional>
#include <stack>
#include <map>
#include <list>
#include <cstdlib>
#include <typeinfo>
#include <typeindex>
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>

#define TIC(KLASS) std::type_index(typeid(KLASS))
#define TIO(OBJECT) std::type_index(typeid(*&*(OBJECT)))

enum class LOG_LEVEL
{
	LL_ERROR,
	LL_INFO,
	LL_DEBUG
};

class Log
{
public:
	Log(LOG_LEVEL ll) : _log_level(ll) {};
	~Log() 
	{
		if (_log_level <= app_log_level) 
		{
			std::cout << _stream.str() << std::endl;
		}
	}
	LOG_LEVEL _log_level = LOG_LEVEL::LL_ERROR;
	std::stringstream _stream;

	static LOG_LEVEL app_log_level;
};

LOG_LEVEL Log::app_log_level;


#define DEBUG Log(LOG_LEVEL::LL_DEBUG)._stream
#define INFO Log(LOG_LEVEL::LL_INFO)._stream
#define ERROR Log(LOG_LEVEL::LL_ERROR)._stream

#define DEFINE_PUBLIC_CTOR(TYPE) TYPE(std::shared_ptr<BaseType> t) : Node({t}) {}
#define DEFINE_DESC(TYPE) std::string description() override { return #TYPE; }
#define DEFINE_MAKE(TYPE) \
	static std::shared_ptr<TYPE> make() \
	{ \
		return std::make_shared<TYPE>(IntType::make()); \
	} 

#define DEFINE_MAKE_BINARY(TYPE) \
	static std::shared_ptr<TYPE> make(std::shared_ptr<Node> a, std::shared_ptr<Node> b) \
	{ \
		return std::make_shared<TYPE>(a, b); \
	}


static size_t global_node_count = 0;

// Types specify the type of data flowing through a wire.
//
// BaseType ::= BaseType | (Type_1,...,Type_n)
// BaseType ::= int (but there could be others)
//
// In actual ngraph, BaseType would encompass tensors. Call this version ScalarFlow or something.

class BaseType : public std::enable_shared_from_this<BaseType>
{
	virtual std::string description() = 0;
};

class IntType : public BaseType
{
public:
	IntType() : BaseType() {}
	static std::shared_ptr<IntType> make() { return std::make_shared<IntType>(); }
	std::string description() final { return "IntType"; }
};

// Wires are the connections in the graph over which data flows. Each wire carries a value of base type;
// a tuples are transmitted over multiple wires.
//
// A wire has shared ownership of its source node. If the wire is deleted and nothing else (e.g., another
// wire or a live local scope) is holding onto a shared pointer to the source node, the source node will
// be deleted as well.
//
// This is closely analogous to what we are currently calling "Output". The key difference is just in the
// ownership/lifetime relation.

class Node;
class Input;
using Nodes = std::vector<std::shared_ptr<Node>>;

class Wire
{
public:
	Wire(std::shared_ptr<Node> source_node, std::shared_ptr<BaseType> wire_type)
		: m_source_node(source_node)
		, m_wire_type(wire_type)
	{}

	std::shared_ptr<BaseType> get_wire_type() { return m_wire_type; }

	void add_dest_input(Input* input)
	{
		m_dest_inputs.insert(input);
	}

	void delete_dest_input(Input* input)
	{
		m_dest_inputs.erase(input);
	}

	std::set<Input*> get_dest_inputs()
	{
		return m_dest_inputs;
	}

	std::shared_ptr<Node> get_source_node()
	{
		return m_source_node;
	}

private:
	// The node that generates the output value for this wire.
	std::shared_ptr<Node> m_source_node;

	// All the inputs that are reading this wire.
	std::set<Input*> m_dest_inputs;

	// The type being carried over the wire. Note that it must be a base type.
	std::shared_ptr<BaseType> m_wire_type;
};

// An Input is an input port for a node. Each node has 0 or more inputs. A node owns its Inputs by
// containment, and an Input has shared ownership (via shared_ptr) of the wire that it reads. More
// than one Input can be reading the same wire.

class Node;

class Input
{
public:
	Input(Node* node, std::shared_ptr<Wire> wire)
		: m_node(node)
		, m_wire(wire)
	{
		//std::cout << "node = " << node << " wire = " << wire << " input = " << this << std::endl;
		wire->add_dest_input(this);
	}

	Input(const Input &ci) : m_node(ci.m_node), m_wire(ci.m_wire)
	{
		m_wire->add_dest_input(this);
	}

	~Input()
	{
		m_wire->delete_dest_input(this);
	}

	static std::shared_ptr<Input> make(Node* node, std::shared_ptr<Wire> wire)
	{
		return std::make_shared<Input>(node, wire);
	}

	static std::shared_ptr<Input> make(std::shared_ptr<Node> node, std::shared_ptr<Wire> wire)
	{
		return make(node.get(), wire);
	}

	void replace_wire(std::shared_ptr<Wire> new_wire) { m_wire = new_wire; }

	std::shared_ptr<Wire> get_wire() { return m_wire; }
private:
	// The node that owns this input, i.e., the node that this input is an input to.
	Node *m_node;

	// The wire that this input is reading from.
	std::shared_ptr<Wire> m_wire;
};

// Nodes are the computation elements in the graph. A node has zero or more input wires, and zero or more
// output wires.

class Node : public std::enable_shared_from_this<Node>
{
public:

	virtual std::vector<std::shared_ptr<Wire>> get_output_wires()
	{
		auto sp_wires = std::vector<std::shared_ptr<Wire>>(m_output_wires.size());
		for (size_t i = 0; i < sp_wires.size(); i++)
		{
			auto sp = m_output_wires[i].lock();
			// If the weak pointer is expired or hasn't been initialized, create a fresh wire. It's up
			// to the caller to keep that wire alive.
			if (nullptr == sp)
			{
				sp = std::make_shared<Wire>(shared_from_this(), m_wire_types[i]);
				m_output_wires[i] = sp;  // Note: assigning shared to weak here.
			}
			sp_wires[i] = sp;
		}
		return sp_wires;
	}

	const std::vector<std::shared_ptr<BaseType>>& get_output_types() { return m_wire_types; }
	std::set<std::shared_ptr<Node>> get_unique_children();
	std::set<std::shared_ptr<Node>> get_unique_parents();
	Nodes get_children();
	virtual bool is_commutative() { return false; }
	std::string to_string() { return description() + std::to_string(m_id); }


	~Node() { std::cout << "Node" << m_id << " goes gentle into that night\n"; }
protected:

	virtual std::string description() = 0;
	Node(const std::vector<std::shared_ptr<BaseType>>& wire_types)
		: m_wire_types(wire_types)
		, m_output_wires(wire_types.size())
		, m_id(global_node_count++)
	{
	}
	// The inputs to this node.
	std::vector<Input> m_inputs;

	// The node's output wires.
	std::vector<std::weak_ptr<Wire>> m_output_wires;

	// The node's output type.
	std::vector<std::shared_ptr<BaseType>> m_wire_types;

	//
	size_t m_id;
};

Nodes Node::get_children()
{
	Nodes nodes{};
	for (auto input : m_inputs)
	{
		nodes.push_back(std::shared_ptr<Node>(input.get_wire()->get_source_node()));
	}
	return nodes;
}

std::set<std::shared_ptr<Node>> Node::get_unique_children()
{
	std::set<std::shared_ptr<Node>> nodes{};
	for (auto input : m_inputs)
	{
		nodes.insert(std::shared_ptr<Node>(input.get_wire()->get_source_node()));
	}
	return nodes;
}

std::set<std::shared_ptr<Node>> Node::get_unique_parents()
{
	std::set<std::shared_ptr<Node>> nodes{};
	for (auto wire : get_output_wires())
	{
		nodes.insert(wire->get_source_node());
	}
	return nodes;
}


bool always_visit_children(std::shared_ptr<Node>)
{
	return true;
}

//visitors take std::shared_ptr, not sure if there's any value for a visitor to not own the node while processing it? 
struct Visitor {
	Visitor(std::function<void(std::shared_ptr<Node>)> vs, std::function<bool(std::shared_ptr<Node>)> vc = always_visit_children) : m_visit_node(vs), m_visit_children(vc) {}
	std::function<void(std::shared_ptr<Node>)> m_visit_node;
	std::function<bool(std::shared_ptr<Node>)> m_visit_children;
};


//This materializes the order. I can't think of any good scenarios for this aside from performance.
struct Collector : public Visitor
{
	std::vector<std::shared_ptr<Node>> m_nodes;
	Collector() : Visitor([this](std::shared_ptr<Node> n) {this->m_nodes.push_back(n); }) {}
};

//This traversal seems to play nice with get_unique_children returning shared_ptr
//Unless a visitor starts deleting parents :-) 
void children_first(std::vector<std::shared_ptr<Node>> nodes_to_walk /*Presumably, we own the list we want to walk*/, Visitor& v)
{
	std::stack<std::shared_ptr<Node>> stack;
	for (auto nsp : nodes_to_walk)
	{
		stack.push(nsp);
	}

	//this is a problem; 
	std::set<std::shared_ptr<Node>> visited;

	while (!stack.empty())
	{
		auto current = stack.top();
		if (visited.count(current))
		{
			v.m_visit_node(current);
			stack.pop();
		}
		else
		{
			visited.insert(current);
			if (v.m_visit_children(current))
			{
				for (auto arg : current->get_unique_children())
				{
					stack.push(arg);
				}
			}
		}
	}
}

/*
based on Adjoints::adjoints' traversal
this in theory could work w/ the shared_ptr approach but it's hard to reason about
*/
void parents_first(std::vector<std::shared_ptr<Node>> nodes_to_walk, Visitor& v)
{
	std::map<std::shared_ptr<Node>, size_t> parent_counts;
	std::set<std::shared_ptr<Node>> visited_nodes;

	std::list<std::shared_ptr<Node>> nodes_to_check;
	for (auto n : nodes_to_walk) //TODO: use std::copy
	{
		nodes_to_check.push_front(n);
	}


	while (nodes_to_check.size() > 0)
	{
		auto node = nodes_to_check.front();
		nodes_to_check.pop_front();
		if (visited_nodes.count(node) != 0)
		{
			continue;
		}

		visited_nodes.insert(node);
		if (!v.m_visit_children(node))
		{
			continue;
		}

		for (auto arg : node->get_unique_children()) //this is probably safe in absense of concurrent walks, otherwise create a copy
		{
			auto count_it = parent_counts.find(arg);
			if (count_it == parent_counts.end())
			{
				parent_counts[arg] = 1;
				nodes_to_check.push_front(arg);
			}
			else
			{
				parent_counts[arg]++;
			}
		}
	}

	for (auto n : nodes_to_walk) //TODO: use std::copy
	{
		nodes_to_check.push_front(n);
	}

	while (nodes_to_check.size() > 0)
	{
		auto node = nodes_to_check.front();
		nodes_to_check.pop_front();
		for (auto arg : node->get_unique_children())
		{
			auto count_it = parent_counts.find(arg);
			count_it->second--;
			if (0 == count_it->second)
			{
				nodes_to_check.push_front(arg);
			}
		}
		v.m_visit_node(node);
	}
}

class Binary : public Node
{
public:
	Binary(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
		: Node({ IntType::make() })
	{
		auto a_wires = a->get_output_wires();
		assert(a_wires.size() == 1);
		assert(std::dynamic_pointer_cast<IntType>(a_wires[0]->get_wire_type()));

		auto b_wires = b->get_output_wires();
		assert(b_wires.size() == 1);
		assert(std::dynamic_pointer_cast<IntType>(b_wires[0]->get_wire_type()));

		m_inputs = std::vector<Input>{ Input(this,a_wires[0]),Input(this,b_wires[0]) };
	}
};

class RnnCell : public Node 
{
public:
	RnnCell(std::shared_ptr<Node> x, std::shared_ptr<Node> w, std::shared_ptr<Node> hx)
		: Node({ IntType::make() }) 
	{
		auto a_wires = x->get_output_wires();
		assert(a_wires.size() == 1);
		assert(std::dynamic_pointer_cast<IntType>(a_wires[0]->get_wire_type()));

		auto b_wires = w->get_output_wires();
		assert(b_wires.size() == 1);
		assert(std::dynamic_pointer_cast<IntType>(b_wires[0]->get_wire_type()));

		auto c_wires = hx->get_output_wires();
		assert(c_wires.size() == 1);
		assert(std::dynamic_pointer_cast<IntType>(c_wires[0]->get_wire_type()));

		m_inputs = std::vector<Input>{ Input(this,a_wires[0]),Input(this,b_wires[0]),Input(this,c_wires[0]) };
	}



	DEFINE_DESC(RnnCell)
	static std::shared_ptr<RnnCell> make(std::shared_ptr<Node> a, std::shared_ptr<Node> b, std::shared_ptr<Node> c) \
	{
		return std::make_shared<RnnCell>(a, b, c);
	}

};

class Sub : public Binary
{
public:
	using Binary::Binary;
	DEFINE_MAKE_BINARY(Sub)
		DEFINE_DESC(Sub)
};

class Add : public Binary
{
public:
	using Binary::Binary;
	DEFINE_MAKE_BINARY(Add)
		DEFINE_DESC(Add)
};

class Variable : public Node
{
public:
	DEFINE_PUBLIC_CTOR(Variable)
		DEFINE_MAKE(Variable)
		DEFINE_DESC(Variable)
};

class Pattern : public Node
{
public:
	DEFINE_PUBLIC_CTOR(Pattern)
		DEFINE_MAKE(Pattern)
		DEFINE_DESC(Pattern)
};

class Constant : public Node
{
public:
	Constant(int value)
		: Node({ IntType::make() })
		, m_value(value)
	{
	}

	static std::shared_ptr<Constant> make(int value)
	{
		return std::make_shared<Constant>(value);
	}

	int get_value() { return m_value; }

	DEFINE_DESC(Constant)
private:
	int m_value;
};

class MakeTuple : public Node
{
public:
	MakeTuple(std::vector<std::shared_ptr<Node>> element_nodes)
		: Node(make_tuple_type_from_elements(element_nodes))
	{
		for (auto n : element_nodes)
		{
			auto n_wires = n->get_output_wires();
			for (auto w : n_wires)
			{
				m_inputs.push_back(Input(this, w));
			}
		}

		// Note that we do not actually take "ownership" of these output wires.
		for (size_t i = 0; i < m_inputs.size(); i++)
		{
			m_output_wires[i] = m_inputs[i].get_wire();  // Assigning shared to weak
		}
	}

	static std::shared_ptr<MakeTuple> make(std::vector<std::shared_ptr<Node>> element_nodes)
	{
		return std::make_shared<MakeTuple>(element_nodes);
	}

	DEFINE_DESC(MakeTuple)

		static std::vector<std::shared_ptr<BaseType>> make_tuple_type_from_elements(const std::vector<std::shared_ptr<Node>>& element_nodes)
	{
		std::vector<std::shared_ptr<BaseType>> result;

		for (auto n : element_nodes)
		{
			for (auto type : n->get_output_types())
				result.push_back(type);
		}

		return result;
	}
};

class GetTupleElement : public Node
{
public:
	GetTupleElement(std::shared_ptr<Node> input_node, size_t index)
		: Node({ input_node->get_output_types().at(index) })
	{
		m_output_wires[0] = input_node->get_output_wires().at(index);
	}

	static std::shared_ptr<GetTupleElement> make(std::shared_ptr<Node> input_node, size_t index)
	{
		return std::make_shared<GetTupleElement>(input_node, index);
	}

	DEFINE_DESC(GetTupleElement)
};


class Function
{
public:
	Function(std::shared_ptr<MakeTuple> mt) : m_results(mt) {}
	std::shared_ptr<MakeTuple> m_results; //always returns a tuple this should allow replace_node replace result nodes uniformly 
										  //regardless if they are regular nodes of they feed into the result tuple
};

class Pass
{
public:
	Pass(Function& f) : m_function(f) {}
	void run();
protected:
	virtual void process_node(std::shared_ptr<Node> n) = 0;
	//single-value nodes
	void replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);
	//multi-value nodes
	void replace_outputs(std::map<std::shared_ptr<Wire>, std::shared_ptr<Wire>> old_to_new) { throw "NYI"; }
private:
	Function& m_function; //pass is meaningless w/o a function
};

void Pass::replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement)
{
	if (target->get_output_wires().size() != 1 || replacement->get_output_wires().size() != 1)
	{
		throw "Use replace_outputs for multi-value nodes!";
	}

	auto target_wire = target->get_output_wires().at(0);
	auto inputs = target->get_output_wires().at(0)->get_dest_inputs(); //this is a copy?
	auto repl_wire = replacement->get_output_wires().at(0);
	//TODO hygiene clean up input_dests on target

	for (auto input : inputs)
	{
		input->replace_wire(repl_wire);
	}
}

void Pass::run()
{
	std::vector<std::shared_ptr<Node>> nodes;
	for (auto child : m_function.m_results->get_unique_children())
	{
		nodes.push_back(child);
	}

	Visitor v([this](std::shared_ptr<Node> n) {
		this->process_node(n);
	});

	children_first(nodes, v);
}

//Non-pattern-matcher-based-arithmetic simplifier
class Simplifier : public Pass
{
	using Pass::Pass;
	void process_node(std::shared_ptr<Node> n) override
	{
		static const auto addc = TIC(Add);
		static const auto constc = TIC(Constant);
		static const auto varc = TIC(Variable);

		if (addc == TIO(n))
		{
			auto children = n->get_unique_children();


			if (children.size() == 2)
			{
				//Jayaram's style :-) 
				std::shared_ptr<Constant> const1;
				std::shared_ptr<Constant> const2;
				std::shared_ptr<Variable> variable1;

				for (auto child : children)
				{
					if (constc == TIO(child))
					{
						if (!const1)
						{
							const1 = std::dynamic_pointer_cast<Constant>(child);
						}
						else
						{
							const2 = std::dynamic_pointer_cast<Constant>(child);
						}
					}
					else if (varc == TIO(child))
					{
						variable1 = std::dynamic_pointer_cast<Variable>(child);
					}
				}

				if (const1 && const2)
				{
					auto new_constant = Constant::make(const1->get_value() + const2->get_value()); //overflow semantic?
					replace_node(n, new_constant); //TODO: in after pass check that n went gently into that night
				}

				if (variable1 && const1 && const1->get_value() == 0)
				{
					//handle case var + 0 
				}
			}
		}
	}
};




class Matcher
{
public:
	Matcher(std::shared_ptr<Node> pattern) : m_pattern(pattern) {}
	std::shared_ptr<Node> m_pattern;
	using PatternMap = std::map<std::shared_ptr<Pattern>, std::shared_ptr<Node>>;
	using RPatternMap = std::map<std::shared_ptr<Pattern>, Nodes>;
	PatternMap m_pattern_map;

	bool match(std::shared_ptr<Node> graph)
	{
		m_pattern_map.clear();
		return match(graph, m_pattern, m_pattern_map);
	}

	static bool match_recurring_pattern(std::shared_ptr<Node> graph, std::shared_ptr<Node> pattern, std::shared_ptr<Pattern> rpattern, RPatternMap& patterns);
private:
	bool virtual match_class(std::shared_ptr<Node> graph, std::shared_ptr<Node> pattern, PatternMap& pattern_map);
	bool match_arguments(const Nodes& graph_args, const Nodes& pattern_args, PatternMap& pattern_map);
	bool match(std::shared_ptr<Node> graph, std::shared_ptr<Node> pattern, PatternMap& pattern_map);
	static std::vector<std::shared_ptr<Node>> get_arguments(std::shared_ptr<Node> n);
};

bool Matcher::match_recurring_pattern(std::shared_ptr<Node> graph, std::shared_ptr<Node> pattern, std::shared_ptr<Pattern> rpattern, RPatternMap& patterns)
{
	bool no_match = false;
	Matcher m(pattern);

	INFO << "matching graph = " << graph->to_string() << std::endl;
	while (m.match(graph)) 
	{
		no_match = true;
		graph = m.m_pattern_map[rpattern]; //for the next round
		INFO << "setting graph = " << graph->to_string() << std::endl;
		for (auto me : patterns)
		{
			patterns[me.first].push_back(m.m_pattern_map[me.first]);
		}

	}

	return no_match;
}

bool Matcher::match_arguments(const Nodes& graph_args, const Nodes& pattern_args, PatternMap& pattern_map)
{
	for (size_t i = 0; i < graph_args.size(); i++)
	{
		if (!match_class(pattern_args.at(i), graph_args.at(i), pattern_map))
		{
			return false;
		}
	}

	return true;
}

std::vector<std::shared_ptr<Node>> Matcher::get_arguments(std::shared_ptr<Node> n)
{
	std::vector<std::shared_ptr<Node>> result{};
	for (auto wpn : n->get_unique_children())
	{
		result.push_back(std::shared_ptr<Node>(wpn));
	}

	return result;
}

//This could be actually transformed into map and default handler so it could be "pluggable" into Matcher to 
//easily customize match_class (i.e. versus overriding the whole match_class we could overload invdividual class cases)
bool Matcher::match_class(std::shared_ptr<Node> graph, std::shared_ptr<Node> pattern, PatternMap& pattern_map)
{

	if (!graph || !pattern)
	{
		return false;
	}

	static const auto constc = TIC(Constant);

	if (auto casted_pattern = std::dynamic_pointer_cast<Pattern>(pattern))
	{
		if (pattern_map.count(casted_pattern))
		{
			return (pattern_map[casted_pattern] == graph);
		}
		else
		{
			pattern_map[casted_pattern] = graph;
			return true;
		}
	}

	//Otherwise compare types
	return TIO(pattern) == TIO(graph);
}

bool Matcher::match(std::shared_ptr<Node> graph, std::shared_ptr<Node> pattern, PatternMap& pattern_map)
{
	if (!match_class(graph, pattern, pattern_map))
	{
		return false;
	}

	auto args = get_arguments(graph);
	auto pattern_args = get_arguments(pattern);

	if (args.size() != pattern_args.size())
	{
		return false;
	}

	bool is_match = false;
	if (graph->is_commutative())
	{
		std::sort(begin(pattern_args), end(pattern_args)); //perms in lexicographical order
		do
		{
			PatternMap tmp{ pattern_map };
			if (match_arguments(pattern_args, args, tmp))
			{
				pattern_map.insert(begin(tmp), end(tmp));
				return true;
			}
		} while (std::next_permutation(begin(pattern_args), end(pattern_args)));
	}
	else
	{
		PatternMap tmp{ pattern_map };
		if (match_arguments(pattern_args, args, tmp)) 
		{
			pattern_map.insert(begin(tmp), end(tmp));
			return true;
		}
	}

	return false;
}

class SimplifierBasedOnMatcher : public Pass
{
public:
	SimplifierBasedOnMatcher(Function& f) : Pass(f)
	{
		m_label = Pattern::make();
		auto pattern = Add::make(m_label, Constant::make(0));
		m_matcher = std::make_unique<Matcher>(pattern);
	}

	void process_node(std::shared_ptr<Node> n)
	{
		if (m_matcher->match(n))
		{
			replace_node(n, m_matcher->m_pattern_map[m_label]);
		}
	}
private:
	std::unique_ptr<Matcher> m_matcher;
	std::shared_ptr<Pattern> m_label;
};

class GraphRewrite
{
	//TODO: implement
};

Function optimizeFunction()
{
	auto n = Add::make(Constant::make(0), Constant::make(1));
	auto m = Add::make(Constant::make(2), Constant::make(3));
	auto t = MakeTuple::make({ n,m });
	auto z = GetTupleElement::make(t, 0);

	Visitor v([](std::shared_ptr<Node> n) { std::cout << n->to_string() << std::endl;  });
	std::cout << "-------\n";
	children_first({ n }, v);
	std::cout << "-------\n";
	children_first({ m }, v);
	std::cout << "-------\n";
	children_first({ t }, v);
	std::cout << "---parents_first---\n";
	parents_first({ t }, v);
	std::cout << "-------\n";
	children_first({ z }, v);

	Function func1{ t };
	Simplifier s1{ func1 };
	s1.run();
	std::cout << "---Simplifier---\n";
	children_first({ t }, v);
	std::cout << "---n,m and constants should die---\n";
	return func1;
}

void test_rnn_matching() 
{
	auto seed = Constant::make(0);
	auto x = Variable::make();
	auto w = Variable::make();

	auto rnn1 = RnnCell::make(x, w, seed);
	DEBUG << "x = " << x->to_string() << " , w = " << w->to_string() << " , rnn1 = " << rnn1->to_string() << std::endl;

	auto x2 = Variable::make();
	auto w2 = Variable::make();

	auto rnn2 = RnnCell::make(x2, w2, rnn1);
	DEBUG << "x2 = " << x2->to_string() << " , w2 = " << w2->to_string() << " , rnn2 = " << rnn2->to_string() << std::endl;

	auto x3 = Variable::make();
	auto w3 = Variable::make();

	auto rnn3 = RnnCell::make(x3, w3, rnn2);
	DEBUG << "x3 = " << x3->to_string() << " , w3 = " << w3->to_string() << " , rnn3 = " << rnn3->to_string() << std::endl;


	auto rpattern = Pattern::make(); //recurring

	auto xpattern = Pattern::make();
	auto wpattern = Pattern::make();

	auto rnn_pattern = RnnCell::make(xpattern, wpattern, rpattern);

	Matcher::RPatternMap map;
	map[xpattern] = Nodes{};
	map[wpattern] = Nodes{};
	bool result = Matcher::match_recurring_pattern(rnn3, rnn_pattern, rpattern, map);
}

int main()
{

	Log::app_log_level = LOG_LEVEL::LL_DEBUG;
	test_rnn_matching();

	{
		auto func1 = optimizeFunction();
		std::cout << "---the rest die---\n";
	}


	{
		auto n = Add::make(Constant::make(0), Constant::make(1));
		auto m = Add::make(Constant::make(2), Constant::make(3));
		auto t = MakeTuple::make({ n,m });
		auto t2 = MakeTuple::make({ t });

		Visitor v([](std::shared_ptr<Node> n) { std::cout << n->to_string() << std::endl;  });
		std::cout << "------\n";
		children_first({ t }, v);
		std::cout << "------\n";
		children_first({ t2 }, v);
		Function func1{ t2 };

		SimplifierBasedOnMatcher sbom1{ func1 };
		sbom1.run();
		std::cout << "---SimplifierBasedOnMatcher---\n";
		children_first({ t2 }, v);
	}

	{
		std::cout << "---Matcher Tests---\n";

		auto n = Add::make(Constant::make(0), Constant::make(1));
		auto m = Add::make(Constant::make(2), Constant::make(3));
		auto t = MakeTuple::make({ n,m });
		auto z = GetTupleElement::make(t, 0);
		Visitor v([](std::shared_ptr<Node> n) { std::cout << n->to_string() << std::endl;  });
		children_first({ z }, v);
		std::cout << "------\n";
		auto n2 = Add::make(Constant::make(0), Constant::make(1));
		auto m2 = Sub::make(Constant::make(2), Constant::make(3));
		auto t2 = MakeTuple::make({ n2,m2 });
		auto z2 = GetTupleElement::make(t2, 1);
		//auto z2 = GetTupleElement::make(t2, 0); //MATCHER RESULT = 1
		children_first({ z2 }, v);

		Matcher matcher{ z2 };

		std::cout << "MATCHER RESULT = " << matcher.match(z) << std::endl;
	}

	return 0;
}


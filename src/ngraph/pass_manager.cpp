
struct Pass2
{
    size_t id;
    Pass2(const std::string& name, PassManager2& manager)
        : id(counter++)
        , m_name(name)
        , m_manager(manager)
    {
    }

    virtual std::set<analysis_kind> get_required_analyses() = 0;
    std::string m_name;

    //default to run over all nodes in sorted order
    //either run or run_on_node can be redefined to change default behaviour
    virtual bool run(Function& f)
    {
        for (auto node : f.get_ordered_ops())
        {
            run_on_node(node);
        }
    };

    virtual bool run_on_node(std::shared_ptr<Node> node) {}
    //a pass can ask pass_manager to run more passes conditionally
    std::list<Pass2> m_post_passes;
    PassManager2& m_manager;
};

//Repeat given passes N times
struct Repeater : public Pass2
{
    Repeater(const std::string& name,
             PassManager2& manager,
             const std::list<Pass2>& passes,
             size_t count)
        : Pass2(name, manager)
        , m_passes(passes)
        , m_count(count)
    {
        if (count < 1)
        {
            throw "count is less than 1";
        }
    }
    std::set<analysis_kind> get_required_analyses() override { return std::set<analysis_kind>{}; }
    bool run()
    {
        bool changed = false;
        while (m_count--)
        {
            changed = m_manager.run_passes(m_passes) || changed;
        }

        return changed;
    }

private:
    std::list<Pass2> m_passes;
    size_t m_count;
};

//Repeat given passes until IR is transforming
struct RepeaterUntil : public Pass2
{
    RepeaterUntil(const std::string& name, PassManager2& manager, const std::list<Pass2>& passes)
        : Pass2(name, manager)
        , m_passes(passes)
    {
    }
    std::set<analysis_kind> get_required_analyses() override { return std::set<analysis_kind>{}; }
    bool run()
    {
        bool changed = false;
        do
        {
            //a hook passes can use to make manager run more passes and take of required analyses
            //and any differences between module/function/node passes
            changed = m_manager.run_passes(m_passes) || changed;
        } while (changed);

        return false;
    }

private:
    std::list<Pass2> m_passes;
    size_t m_count;
};

//Group passes into group (core,backend-specific,etc)
struct GroupPass : public Repeater
{
    GroupPass(const std::string& name, PassManager2& manager, const std::list<Pass2>& passes)
        : Repeater(name, manager, passes, 1)
    {
    }
};

class AnalysisData
{
};

struct Analysis2
{
    std::unique_ptr<AnalysisData> run(Function f) { return nullptr; }
};

struct PassManager2
{
    PassManager2(Function& f)
        : m_function(f)
    {
    }

    //available analyses; backends should be able to register more analyses
    std::map<analysis_kind, std::unique_ptr<Analysis2>> analyses;

    //API passes use to invalidate the result of analyses
    void invalidate_analysis(analysis_kind ak) { valid_analyses.erase(ak); }
    //can be used for instrumentation
    std::list<Pass2> m_pre_passes;
    std::list<Pass2> m_post_passes;

    //a backend or optimizer register availablbe analysis types
    void register_analysis(analysis_kind, const std::unique_ptr<Analysis2>& analysis) {}
    std::map<analysis_kind, std::unique_ptr<AnalysisData>> valid_analyses;

    Function& m_function;
    bool run_passes(const std::list<Pass2>& passes)
    {
        bool changed = false;
        for (auto pass : passes)
        {
            //instrumentation
            for (auto ipass : m_pre_passes)
            {
                //should we be able to bail out on a pass?
                ipass.run(m_function);
            }
            auto akinds = pass.get_required_analyses();
            for (auto ak : akinds)
            {
                if (valid_analyses.count(ak) == 0)
                {
                    auto adata = analyses.at(ak)->run(m_function);
                    valid_analyses[ak] = std::move(adata);
                }
            }
            changed = pass.run(m_function) || changed;
            if (pass.m_post_passes.size() != 0)
            {
                //it's okay to do this recursively since we won't have
                //a very long dependency tree practically
                changed = run_passes(pass.m_post_passes) || changed;
            }
            //post passes instrumentation
            for (auto ipass : m_post_passes)
            {
                ipass.run(m_function);
            }
        }

        return changed;
    }
};

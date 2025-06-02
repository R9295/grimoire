use std::{io::ErrorKind, path::PathBuf, process::Command, time::Duration};

use clap::{Parser};
use libafl::{
    corpus::{CachedOnDiskCorpus, OnDiskCorpus},
    events::{ClientDescription, EventConfig, Launcher},
    executors::ForkserverExecutor,
    feedback_or, feedback_or_fast,
    feedbacks::{CrashFeedback, MaxMapFeedback, TimeFeedback},
    inputs::{BytesInput, GeneralizedInputMetadata},
    monitors::MultiMonitor,
    mutators::{
        havoc_mutations, tokens_mutations, GrimoireExtensionMutator, GrimoireRandomDeleteMutator,
        GrimoireRecursiveReplacementMutator, GrimoireStringReplacementMutator,
        HavocScheduledMutator, Tokens,
    },
    observers::{CanTrack, HitcountsMapObserver, StdMapObserver, TimeObserver},
    schedulers::{powersched::PowerSchedule, StdWeightedScheduler},
    stages::{GeneralizationStage, StdMutationalStage},
    state::StdState,
    Fuzzer, HasMetadata, StdFuzzer,
};
use libafl_bolts::{
    core_affinity::Cores,
    current_nanos,
    rands::StdRand,
    shmem::{ShMem, ShMemProvider, StdShMemProvider, UnixShMemProvider},
    tuples::{tuple_list, Merge},
    AsSliceMut, TargetArgs,
};
const SHMEM_ENV_VAR: &str = "__AFL_SHM_ID";
fn main() {
    let opt = Opt::parse();
    let shmem_provider = StdShMemProvider::new().expect("Failed to init shared memory");
    let monitor = MultiMonitor::new(|s| println!("{s}"));
    match std::fs::create_dir(&opt.out_dir) {
        Ok(_) => {}
        Err(e) => {
            if !matches!(e.kind(), ErrorKind::AlreadyExists) {
                panic!("{:?}", e)
            }
        }
    };
    if !opt.out_dir.join("queue").exists() {
        std::fs::create_dir(opt.out_dir.join("queue")).unwrap();
    }
    if !opt.out_dir.join("crashes").exists() {
        std::fs::create_dir(opt.out_dir.join("crashes")).unwrap();
    }
    let run_client = |mut state: Option<_>,
                      mut mgr: _,
                      core: ClientDescription|
     -> Result<(), libafl_bolts::Error> {
        let map_size = {
            let map_size = Command::new(opt.executable.clone())
                .env("AFL_DUMP_MAP_SIZE", "1")
                .output()
                .expect("target gave no output");
            let map_size = String::from_utf8(map_size.stdout)
                .expect("target returned illegal mapsize")
                .replace("\n", "");
            map_size.parse::<usize>().expect("illegal mapsize output") + opt.map_bias
        };
        // Create the shared memory map for comms with the forkserver
        let mut shmem_provider = UnixShMemProvider::new().unwrap();
        let mut shmem = shmem_provider.new_shmem(map_size).unwrap();
        unsafe {
            shmem.write_to_env(SHMEM_ENV_VAR).unwrap();
        }
        let shmem_buf = shmem.as_slice_mut();
        let edges_observer = unsafe {
            HitcountsMapObserver::new(StdMapObserver::new("edges", shmem_buf))
                .track_indices()
                .track_novelties()
        };
        let map_feedback = MaxMapFeedback::new(&edges_observer);
        // Create an observation channel to keep track of the execution time.
        let time_observer = TimeObserver::new("time");
        let mut feedback = feedback_or!(map_feedback, TimeFeedback::new(&time_observer));
        let mut objective = CrashFeedback::new();
        // Initialize our State if necessary
        let mut state = state.unwrap_or_else(|| {
            StdState::new(
                StdRand::with_seed(current_nanos()),
                // TODO: configure testcache size
                CachedOnDiskCorpus::<BytesInput>::new(opt.out_dir.join("queue"), 1000).unwrap(),
                OnDiskCorpus::<BytesInput>::new(opt.out_dir.join("crashes")).unwrap(),
                &mut feedback,
                &mut objective,
            )
            .unwrap()
        });
        if let Some(dict) = opt.dict_path {
            let mut tokens = Tokens::new();
            tokens = tokens.add_from_files(vec![dict]).expect("tokens");
            state.add_metadata(tokens);
        }
        let scheduler = StdWeightedScheduler::with_schedule(
            &mut state,
            &edges_observer,
            Some(PowerSchedule::explore()),
        );
        // A fuzzer with feedbacks and a corpus scheduler
        let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

        let generalization = GeneralizationStage::new(&edges_observer);
        // Setup a mutational stage with a basic bytes mutator
        let mutator = HavocScheduledMutator::with_max_stack_pow(
            havoc_mutations().merge(tokens_mutations()),
            3,
        );
        let grimoire_mutator = HavocScheduledMutator::with_max_stack_pow(
            tuple_list!(
                GrimoireExtensionMutator::new(),
                GrimoireRecursiveReplacementMutator::new(),
                GrimoireStringReplacementMutator::new(),
                GrimoireRandomDeleteMutator::new(),
            ),
            3,
        );
        let mut stages = tuple_list!(
            generalization,
            StdMutationalStage::new(mutator),
            StdMutationalStage::<_, _, GeneralizedInputMetadata, BytesInput, _, _, _>::transforming(
                grimoire_mutator
            )
        );
        let mut executor = ForkserverExecutor::builder()
            .program(opt.executable.clone())
            .coverage_map_size(map_size)
            .debug_child(true)
            .is_persistent(true)
            .is_deferred_frksrv(true)
            .timeout(Duration::from_millis(opt.hang_timeout * 1000))
            .shmem_provider(&mut shmem_provider)
            .build_dynamic_map(edges_observer, tuple_list!(time_observer))
            .unwrap();
        if state.must_load_initial_inputs() {
            state.load_initial_inputs(
                &mut fuzzer,
                &mut executor,
                &mut mgr,
                &[opt.out_dir.join("queue").clone(), opt.input_dir],
            )?;
        }
        fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)?;
        Ok(())
    };
    let _res = Launcher::builder()
        .cores(&opt.cores)
        .monitor(monitor)
        .run_client(run_client)
        .broker_port(opt.broker_port)
        .shmem_provider(shmem_provider)
        .configuration(EventConfig::from_name("default"))
        .build()
        .launch();
}
#[derive(Debug, Parser, Clone)]
#[command(
    name = "grimoire",
    about = "grimoire",
    author = "aarnav <aarnavbos@gmail.com>"
)]
struct Opt {
    executable: PathBuf,
    #[arg(short = 'o')]
    out_dir: PathBuf,
    #[arg(short = 'c', value_parser=Cores::from_cmdline)]
    cores: Cores,
    #[arg(short = 'i')]
    input_dir: PathBuf,
    /// broker port
    #[arg(short = 'p', default_value_t = 4000)]
    broker_port: u16,
    /// Timeout in seconds
    #[arg(short = 't', default_value_t = 1)]
    hang_timeout: u64,
    /// tokens
    #[arg(short = 'x')]
    dict_path: Option<PathBuf>,
    /// AFL_DUMP_MAP_SIZE + x where x = map bias
    #[arg(short = 'm')]
    map_bias: usize,
}
